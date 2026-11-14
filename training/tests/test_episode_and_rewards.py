"""Episode loop and reward logic, exercised with scripted fake models (no GPU needed)."""

from collections.abc import Sequence

from tutor_rl.config import EpisodeConfig, RewardConfig, load_config
from tutor_rl.episode import Message, Problem, run_episode
from tutor_rl.prompts import END_TOKEN
from tutor_rl.rewards import extract_answer, group_advantages, score_episode, solve_rate


class Scripted:
    """Returns queued replies in order; `n` copies of the next reply when n > 1."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def generate(self, messages: Sequence[Message], *, n: int = 1) -> list[str]:
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        return [reply] * n


PROBLEM = Problem(id="p1", question="What is 6 * 12?", answer="72")


def test_tutor_can_end_episode_and_student_is_tested_after() -> None:
    student = Scripted("I don't know how to start.", "Is it 72?", "6*12 = \\boxed{72}")
    tutor = Scripted("What is 6 * 10 first?", f"Great reasoning! {END_TOKEN}")
    ep = run_episode(PROBLEM, tutor, student, EpisodeConfig(max_turns=8, student_final_attempts=2))

    assert [t.speaker for t in ep.turns] == ["student", "tutor", "student", "tutor"]
    assert ep.ended_by_tutor and END_TOKEN not in ep.turns[-1].content
    assert ep.final_attempts == ["6*12 = \\boxed{72}"] * 2
    # The tutor sees its own messages as assistant turns.
    assert [m.role for m in ep.tutor_view()] == ["system", "user", "assistant", "user", "assistant"]


def test_episode_stops_at_max_turns() -> None:
    student = Scripted("hi", "hmm", "\\boxed{1}")
    tutor = Scripted("hint")
    ep = run_episode(PROBLEM, tutor, student, EpisodeConfig(max_turns=3, student_final_attempts=1))
    assert len(ep.turns) == 3 and not ep.ended_by_tutor


def test_answer_extraction_and_solve_rate() -> None:
    assert extract_answer("so \\boxed{1,200.0}") == "1200"
    assert extract_answer("no box") is None
    assert solve_rate(["\\boxed{72}", "\\boxed{70}"], "72") == 0.5


def _episode(final: str):
    student = Scripted("help", final)
    return run_episode(
        PROBLEM, Scripted(END_TOKEN), student, EpisodeConfig(student_final_attempts=1)
    )


def test_reward_is_solve_rate_when_judge_accepts() -> None:
    judge = Scripted('{"reasoning": "fine", "decision": "OK"}')
    score = score_episode(_episode("\\boxed{72}"), judge, RewardConfig())
    assert (score.reward, score.leaked) == (1.0, False)


def test_leak_gets_minimum_reward_and_bad_json_counts_as_leak() -> None:
    cfg = RewardConfig(leak_penalty=1.0, end_on_rejection=True)
    rejected = score_episode(_episode("\\boxed{72}"), Scripted('{"decision": "REJECT"}'), cfg)
    assert rejected.reward == -1
    assert score_episode(_episode("\\boxed{72}"), Scripted("not json"), cfg).leaked


def test_group_advantages_are_centered() -> None:
    adv = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(adv)) < 1e-9 and adv[0] > 0 > adv[1]
    assert group_advantages([0.5, 0.5]) == [0.0, 0.0]


def test_small_config_loads() -> None:
    cfg = load_config("configs/small.toml")
    assert cfg.episode.max_turns == 8 and cfg.reward.leak_penalty == 1.0
