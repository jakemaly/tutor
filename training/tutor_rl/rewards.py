"""PedagogicalRL-style reward: student success after the dialogue, minus a leak penalty.

Matches the authors' "hard lambda" setting: if the judge rejects the dialogue, the episode gets
the minimum reward (-leak_penalty); otherwise reward = solve rate (- penalty if not ending early).
"""

import json
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from tutor_rl.config import RewardConfig
from tutor_rl.episode import ChatModel, Episode, Message, generate_many
from tutor_rl.prompts import LEAK_JUDGE

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_answer(text: str) -> str | None:
    """Last \\boxed{...} value, normalized; None if the student gave no boxed answer."""
    matches = _BOXED.findall(text)
    return normalize_answer(matches[-1]) if matches else None


def normalize_answer(value: str) -> str:
    value = value.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return value.lower()
    return str(int(number)) if number.is_integer() else str(number)


def solve_rate(attempts: Sequence[str], answer: str) -> float:
    if not attempts:
        return 0.0
    target = normalize_answer(answer)
    return sum(extract_answer(a) == target for a in attempts) / len(attempts)


def judge_prompt(episode: Episode) -> list[Message]:
    return [Message("user", LEAK_JUDGE.format(transcript=episode.transcript()))]


def judge_rejects(episode: Episode, judge: ChatModel) -> bool:
    """True if the judge says the tutor leaked the answer. Unparseable output counts as REJECT."""
    return parse_rejection(judge.generate(judge_prompt(episode))[0])


def parse_rejection(raw: str) -> bool:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        decision = json.loads(match.group(0))["decision"] if match else "REJECT"
    except (json.JSONDecodeError, KeyError, TypeError):
        decision = "REJECT"
    return str(decision).strip().upper() != "OK"


@dataclass(frozen=True)
class EpisodeScore:
    reward: float
    solve_rate: float
    leaked: bool


def score_episode(episode: Episode, judge: ChatModel, cfg: RewardConfig) -> EpisodeScore:
    return score_episodes([episode], judge, cfg)[0]


def score_episodes(
    episodes: Sequence[Episode], judge: ChatModel, cfg: RewardConfig
) -> list[EpisodeScore]:
    """Score several episodes; the judge sees them in one batch."""
    verdicts = generate_many(judge, [judge_prompt(ep) for ep in episodes])
    return [
        _score(ep, parse_rejection(v[0]), cfg) for ep, v in zip(episodes, verdicts, strict=True)
    ]


def _score(episode: Episode, leaked: bool, cfg: RewardConfig) -> EpisodeScore:
    rate = solve_rate(episode.final_attempts, episode.problem.answer)
    if leaked:
        reward = -cfg.leak_penalty if cfg.end_on_rejection else rate - cfg.leak_penalty
    else:
        reward = rate
    return EpisodeScore(reward=reward, solve_rate=rate, leaked=leaked)


def group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> list[float]:
    """GRPO advantage: each reward relative to its group (same problem) mean and std."""
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return [(r - mean) / (std + eps) for r in rewards]


def summarize(episodes: Sequence[Episode], scores: Sequence[EpisodeScore]) -> dict:
    """Mean reward, student solve rate, leak rate and dialogue length over episodes."""
    return {
        "reward": statistics.fmean(s.reward for s in scores),
        "solve_rate": statistics.fmean(s.solve_rate for s in scores),
        "leak_rate": statistics.fmean(s.leaked for s in scores),
        "mean_turns": statistics.fmean(len(e.turns) for e in episodes),
        "ended_by_tutor_rate": statistics.fmean(e.ended_by_tutor for e in episodes),
        "episodes": len(episodes),
    }
