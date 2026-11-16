"""PedagogicalRL reward (paper section 4.1 and appendix B).

    r_sol  = share of the K post-dialog student solutions that are correct
    r_ped  = 1 if every judge sample accepts the dialogue, else 0
             (judges: answer leakage and helpfulness, each sampled twice)
    r      = r_sol + (r_ped - 1) * lam + r_templ            (soft lambda)
    r      = -lam if r_ped == 0, else r_sol + r_templ       (hard lambda)
    r_templ = 0.1 if the tutor ended the dialogue - 0.5 if any tutor turn was truncated

Thinking tags are off for the Qwen3.5 tutor, so the think-tag terms of r_templ are always 0.
"""

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from tutor_rl.config import RewardConfig
from tutor_rl.episode import ChatModel, Episode, Message, generate_many
from tutor_rl.prompts import JUDGES

# ----- answer checking ----------------------------------------------------------------------


def last_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...} (nested braces allowed); None if there is none."""
    start = text.rfind("\\boxed")
    while start != -1:
        i = text.find("{", start)
        if i != -1 and text[start + len("\\boxed") : i].strip() == "":
            depth = 0
            for j in range(i, len(text)):
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                if depth == 0:
                    return text[i + 1 : j].strip()
        start = text.rfind("\\boxed", 0, start)
    return None


def normalize_answer(value: str) -> str:
    """Plain-text fallback used when math-verify cannot parse an answer."""
    value = value.strip().replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return value.lower()
    return str(int(number)) if number.is_integer() else str(number)


_MAX_ANSWER_CHARS = 300  # sympy can stall on huge expressions; nothing real is this long


def answers_match(predicted: str, gold: str) -> bool:
    """Symbolic equivalence with math-verify (what Big-Math was graded with), else string match.

    Timeouts are off: math-verify's timeouts use signals, which only work on the main thread.
    """
    if len(predicted) > _MAX_ANSWER_CHARS:
        return False
    if normalize_answer(predicted) == normalize_answer(gold):
        return True
    try:
        from math_verify import parse, verify
    except ImportError:
        return False
    try:
        g = parse(f"${gold}$", parsing_timeout=None)
        p = parse(f"${predicted}$", parsing_timeout=None)
        return bool(g and p and verify(g, p, timeout_seconds=None))
    except Exception:  # noqa: BLE001 - a grading failure counts as wrong, never crashes a run
        return False


def is_correct(attempt: str, gold: str) -> bool:
    boxed = last_boxed(attempt)
    return boxed is not None and answers_match(boxed, gold)


def solve_rate(attempts: Sequence[str], answer: str) -> float:
    if not attempts:
        return 0.0
    return sum(is_correct(a, answer) for a in attempts) / len(attempts)


# ----- judges --------------------------------------------------------------------------------


def judge_prompt(episode: Episode, kind: str) -> list[Message]:
    """The paper's judge prompt with the conversation filled in, as one user message."""
    return [Message("user", JUDGES[kind].format(conversation=episode.transcript()))]


_DECISION = re.compile(r'"decision"\s*:\s*"?\s*(OK|REJECT)', re.IGNORECASE)


def parse_rejection(raw: str) -> bool:
    """REJECT unless the judge's (last) decision is OK; unparseable output counts as REJECT.

    Reads the "decision" field directly rather than json.loads, because the reasoning often
    quotes LaTeX, whose backslashes are not valid JSON.
    """
    matches = _DECISION.findall(raw)
    return not matches or matches[-1].upper() != "OK"


def run_judges(
    episodes: Sequence[Episode], judge: ChatModel, kinds: Sequence[str], samples: int
) -> list[dict[str, list[str]]]:
    """Per episode: {judge kind: [raw outputs]}; all requests of a kind go out as one batch."""
    out: list[dict[str, list[str]]] = [{} for _ in episodes]
    for kind in kinds:
        replies = generate_many(judge, [judge_prompt(ep, kind) for ep in episodes], n=samples)
        for row, reply in zip(out, replies, strict=True):
            row[kind] = list(reply)
    return out


# ----- scoring -------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeScore:
    reward: float
    solve_rate: float  # r_sol
    accepted: bool  # r_ped == 1
    template: float  # r_templ
    rejected_by: dict[str, bool] = field(default_factory=dict)  # kind -> any sample rejected
    judge_outputs: dict[str, list[str]] = field(default_factory=dict)

    @property
    def leaked(self) -> bool:
        """The paper's "leaked solution": the leakage judge rejected the dialogue."""
        return self.rejected_by.get("leakage", False)


def template_reward(episode: Episode, cfg: RewardConfig) -> float:
    if not cfg.template:
        return 0.0
    bonus = cfg.end_bonus if episode.ended_by_tutor else 0.0
    penalty = cfg.length_penalty if any(s.truncated for s in episode.tutor_steps) else 0.0
    return bonus - penalty


def combine(r_sol: float, accepted: bool, r_templ: float, cfg: RewardConfig) -> float:
    if cfg.hard and not accepted:
        return -cfg.lam
    return r_sol + (float(accepted) - 1.0) * cfg.lam + r_templ


def score_episodes(
    episodes: Sequence[Episode], judge: ChatModel, cfg: RewardConfig
) -> list[EpisodeScore]:
    """Score several episodes; each judge prompt goes to the judge as one batch."""
    verdicts = run_judges(episodes, judge, cfg.judges, cfg.samples_per_judge)
    scores = []
    for ep, outputs in zip(episodes, verdicts, strict=True):
        rejected = {k: any(parse_rejection(o) for o in v) for k, v in outputs.items()}
        accepted = not any(rejected.values())
        r_sol = solve_rate(ep.final_attempts, ep.problem.answer)
        r_templ = template_reward(ep, cfg)
        scores.append(
            EpisodeScore(
                reward=combine(r_sol, accepted, r_templ, cfg),
                solve_rate=r_sol,
                accepted=accepted,
                template=r_templ,
                rejected_by=rejected,
                judge_outputs=outputs,
            )
        )
    return scores


def score_episode(episode: Episode, judge: ChatModel, cfg: RewardConfig) -> EpisodeScore:
    return score_episodes([episode], judge, cfg)[0]


def score_dict(score: EpisodeScore) -> dict:
    return {**vars(score), "leaked": score.leaked}


def group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> list[float]:
    """GRPO advantage: each reward relative to its group (same problem) mean and std."""
    mean = statistics.fmean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0  # unbiased, as in TRL
    return [(r - mean) / (std + eps) for r in rewards]


def summarize(episodes: Sequence[Episode], scores: Sequence[EpisodeScore]) -> dict:
    """Means over episodes of the reward, its parts, and dialogue shape."""
    out = {
        "reward": statistics.fmean(s.reward for s in scores),
        "solve_rate": statistics.fmean(s.solve_rate for s in scores),
        "accept_rate": statistics.fmean(s.accepted for s in scores),
        "template_reward": statistics.fmean(s.template for s in scores),
        "mean_turns": statistics.fmean(len(e.turns) for e in episodes),
        "mean_tutor_words": statistics.fmean(
            statistics.fmean(
                [len(t.content.split()) for t in e.turns if t.speaker == "tutor"] or [0]
            )
            for e in episodes
        ),
        "ended_by_tutor_rate": statistics.fmean(e.ended_by_tutor for e in episodes),
        "truncated_rate": statistics.fmean(
            any(s.truncated for s in e.tutor_steps) for e in episodes
        ),
        "tutor_initiated_rate": statistics.fmean(e.initiator == "tutor" for e in episodes),
        "episodes": len(episodes),
    }
    for kind in sorted({k for s in scores for k in s.rejected_by}):
        out[f"{kind}_reject_rate"] = statistics.fmean(
            s.rejected_by.get(kind, False) for s in scores
        )
    out["leak_rate"] = out.get("leakage_reject_rate", 0.0)
    return out
