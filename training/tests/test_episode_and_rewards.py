"""Episode loop and the paper's reward, exercised with scripted fake models (no GPU needed)."""

import random
from collections.abc import Sequence
from pathlib import Path

import pytest

from tutor_rl.config import EpisodeConfig, RewardConfig, load_config
from tutor_rl.episode import (
    Episode,
    Message,
    Problem,
    Turn,
    TutorStep,
    pick_initiators,
    run_episode,
    run_episodes,
)
from tutor_rl.prompts import END_TOKEN, JUDGE_LEAKAGE, STUDENT_FINAL, TUTOR_SYSTEM
from tutor_rl.rewards import (
    combine,
    group_advantages,
    is_correct,
    judge_prompt,
    last_boxed,
    parse_rejection,
    score_episodes,
    solve_rate,
    template_reward,
)

CONFIGS = Path(__file__).parents[1] / "configs"
OK = '{"reasoning": "fine", "decision": "OK"}'
REJECT = '{"reasoning": "gave it away", "decision": "REJECT"}'


class Scripted:
    """Returns queued replies in order; `n` copies of the next reply when n > 1."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def generate(self, messages: Sequence[Message], *, n: int = 1) -> list[str]:
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        return [reply] * n


class Judge:
    """Answers each judge prompt kind with a fixed list of samples."""

    def __init__(self, leakage: list[str], helpfulness: list[str]) -> None:
        self.by_kind = {"leakage": leakage, "helpfulness": helpfulness}
        self.prompts: list[str] = []

    def generate(self, messages: Sequence[Message], *, n: int = 1) -> list[str]:
        text = messages[0].content
        self.prompts.append(text)
        kind = "leakage" if "revealed too much information" in text else "helpfulness"
        assert len(self.by_kind[kind]) == n
        return self.by_kind[kind]


PROBLEM = Problem(id="p1", question="What is 6 * 12?", answer="72")


def test_student_initiated_dialogue_ends_and_student_is_tested_after() -> None:
    student = Scripted("I don't know how to start.", "Is it 72?", "6*12 = \\boxed{72}")
    tutor = Scripted("What is 6 * 10 first?", f"Great reasoning! {END_TOKEN}")
    ep = run_episode(PROBLEM, tutor, student, EpisodeConfig(max_turns=8, student_final_attempts=2))

    assert [t.speaker for t in ep.turns] == ["student", "tutor", "student", "tutor"]
    assert ep.ended_by_tutor and END_TOKEN not in ep.turns[-1].content
    assert ep.final_attempts == ["6*12 = \\boxed{72}"] * 2
    # The tutor sees its own messages as assistant turns, under the paper's system prompt.
    view = ep.tutor_view()
    assert [m.role for m in view] == ["system", "user", "assistant", "user", "assistant"]
    assert view[0].content == TUTOR_SYSTEM.format(problem=PROBLEM.question)
    # The post-dialog test is the paper's Figure 10 text after the student's chat history.
    final = student.calls[-1]
    assert final[-1] == Message("user", STUDENT_FINAL) and final[0].role == "system"


def test_tutor_initiated_dialogue_starts_from_system_prompt_only() -> None:
    student = Scripted("I think it's 70?", "\\boxed{72}")
    tutor = Scripted("How would you start?", f"Check 6 * 2 again. {END_TOKEN}")
    cfg = EpisodeConfig(max_turns=8, student_final_attempts=1)
    ep = run_episodes([PROBLEM], tutor, student, cfg, initiators=["tutor"])[0]
    assert [t.speaker for t in ep.turns] == ["tutor", "student", "tutor"]
    assert [m.role for m in tutor.calls[0]] == ["system"]
    assert ep.tutor_steps[0].context_turns == 0 and ep.initiator == "tutor"


def test_mixed_openers_run_in_lockstep_up_to_max_turns() -> None:
    cfg = EpisodeConfig(max_turns=3, student_final_attempts=1)
    student = Scripted("s1", "s2", "s3", "\\boxed{1}", "\\boxed{1}")
    tutor = Scripted("t1", "t2", "t3")
    eps = run_episodes([PROBLEM, PROBLEM], tutor, student, cfg, initiators=["student", "tutor"])
    assert [t.speaker for t in eps[0].turns] == ["student", "tutor", "student"]
    assert [t.speaker for t in eps[1].turns] == ["tutor", "student", "tutor"]
    assert not any(e.ended_by_tutor for e in eps)


def test_initiator_is_one_coin_flip_per_problem() -> None:
    picks = pick_initiators(400, EpisodeConfig(initiator="random"), random.Random(0))
    assert set(picks) == {"student", "tutor"} and 150 < picks.count("tutor") < 250
    assert pick_initiators(3, EpisodeConfig(initiator="tutor"), random.Random(0)) == ["tutor"] * 3
    with pytest.raises(ValueError):
        pick_initiators(1, EpisodeConfig(initiator="judge"), random.Random(0))


def test_boxed_extraction_handles_nesting_and_takes_the_last() -> None:
    assert last_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{3} or rather \boxed {4}") == "4"
    assert last_boxed("no box, just 72") is None
    assert not is_correct("the answer is 72", "72")  # the paper's prompt requires \boxed{}
    assert solve_rate([r"\boxed{72}", r"\boxed{70}", "72"], "72") == pytest.approx(1 / 3)
    assert is_correct(r"\boxed{1,200}", "1200")


def test_latex_answers_are_compared_symbolically() -> None:
    pytest.importorskip("math_verify")
    assert is_correct(r"\boxed{0.5}", r"\frac{1}{2}")
    assert is_correct(r"\boxed{\sqrt{12}}", r"2\sqrt{3}")
    assert is_correct(r"\boxed{\dfrac{\pi}{3}}", r"\frac{\pi}{3}")
    assert not is_correct(r"\boxed{66}", "67")


def _scored(final: str, judge: Judge, cfg: RewardConfig, *, ended=True, truncated=False):
    ep = Episode(PROBLEM, turns=[Turn("student", "help"), Turn("tutor", "Try 6*10.")])
    ep.ended_by_tutor = ended
    ep.tutor_steps = [TutorStep(1, "Try 6*10.", truncated)]
    ep.final_attempts = [final, r"\boxed{0}"]
    return score_episodes([ep], judge, cfg)[0]


def test_soft_lambda_reward_with_template_terms() -> None:
    cfg = RewardConfig(lam=0.75)
    accepted = _scored(r"\boxed{72}", Judge([OK, OK], [OK, OK]), cfg)
    assert accepted.solve_rate == 0.5 and accepted.accepted
    assert accepted.reward == pytest.approx(0.5 + 0.1)  # r_sol + end-of-conversation bonus
    # One rejecting sample of either judge is enough to lose r_ped (paper: all must accept).
    leak = _scored(r"\boxed{72}", Judge([OK, REJECT], [OK, OK]), cfg)
    assert leak.leaked and not leak.accepted
    assert leak.reward == pytest.approx(0.5 - 0.75 + 0.1)
    unhelpful = _scored(r"\boxed{72}", Judge([OK, OK], [REJECT, OK]), cfg, ended=False)
    assert not unhelpful.leaked and unhelpful.rejected_by == {"leakage": False, "helpfulness": True}
    assert unhelpful.reward == pytest.approx(0.5 - 0.75)
    cut = _scored(r"\boxed{72}", Judge([OK, OK], [OK, OK]), cfg, ended=False, truncated=True)
    assert cut.template == -0.5 and cut.reward == pytest.approx(0.0)


def test_hard_lambda_sets_a_fixed_penalty() -> None:
    cfg = RewardConfig(lam=1.0, hard=True)
    assert _scored(r"\boxed{72}", Judge([REJECT, REJECT], [OK, OK]), cfg).reward == -1.0
    assert _scored(r"\boxed{72}", Judge([OK, OK], [OK, OK]), cfg).reward == pytest.approx(0.6)
    assert combine(1.0, False, 0.1, RewardConfig(lam=0.0)) == pytest.approx(1.1)


def test_template_reward_can_be_turned_off() -> None:
    ep = Episode(PROBLEM, ended_by_tutor=True)
    assert template_reward(ep, RewardConfig()) == pytest.approx(0.1)
    assert template_reward(ep, RewardConfig(template=False)) == 0.0


def test_judge_gets_the_paper_prompt_with_teacher_and_student_lines() -> None:
    ep = Episode(PROBLEM, turns=[Turn("student", "help"), Turn("tutor", "What is 6*10?")])
    text = judge_prompt(ep, "leakage")[0].content
    conversation = "- Student: help\n- Teacher: What is 6*10?"
    assert text == JUDGE_LEAKAGE.format(conversation=conversation)
    assert f"==BEGIN CONVERSATION==\n{conversation}\n==END CONVERSATION==" in text
    judge = Judge([OK, OK], [OK, OK])
    _scored(r"\boxed{72}", judge, RewardConfig())
    assert len(judge.prompts) == 2  # one request per judge prompt, two samples each


def test_judge_decision_parsing_tolerates_latex_and_takes_last_decision() -> None:
    latex = r'{"reasoning": "The tutor wrote \(16 - 3 = 13\).", "decision": "REJECT"}'
    assert parse_rejection(latex)  # not valid JSON, still read as REJECT
    assert not parse_rejection('```json\n{"reasoning": "only hints", "decision": "OK"}\n```')
    assert parse_rejection('first "decision": "OK" then corrected: "decision": "REJECT"')
    assert parse_rejection("I think it is fine.")  # no decision -> REJECT


def test_group_advantages_are_centered() -> None:
    adv = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(adv)) < 1e-9 and adv[0] > 0 > adv[1]
    assert group_advantages([0.5, 0.5]) == [0.0, 0.0]


def test_molab_config_has_the_papers_values() -> None:
    cfg = load_config(CONFIGS / "molab.toml")
    t = cfg.train
    assert (t["problems_per_step"], t["episodes_per_problem"], t["num_iterations"]) == (16, 8, 2)
    assert t["max_steps"] * t["num_iterations"] == 300 and t["beta"] == 0.001
    assert cfg.episode == EpisodeConfig(max_turns=16, student_final_attempts=8, initiator="random")
    assert cfg.reward.judges == ("leakage", "helpfulness") and cfg.reward.samples_per_judge == 2
    assert cfg.tutor["temperature"] == 1.0 and cfg.eval["problems"] == 500
    assert cfg.data["dataset"] == "rd211/Big-Math-RL-Verified-Filtered"
    assert (cfg.data["train_limit"], cfg.data["eval_limit"]) == (10000, 500)
    smoke = load_config(CONFIGS / "smoke.toml")
    assert smoke.tutor == {**cfg.tutor, "max_tokens_per_turn": 128, "generation_batch_size": 16}
    assert (smoke.student, smoke.judge, smoke.reward) == (cfg.student, cfg.judge, cfg.reward)


def test_gpu_budget_keeps_everything_under_80_percent() -> None:
    from dataclasses import replace

    from tutor_rl.gpu_budget import plan

    for name in ("molab", "smoke"):
        cfg = load_config(CONFIGS / f"{name}.toml")
        train, evaluating = plan(cfg), plan(cfg, evaluating=True)
        assert cfg.gpu["max_fraction"] == 0.80
        for budget in (train, evaluating):
            assert sum(budget.servers.values()) + budget.tutor == pytest.approx(0.80)
        assert set(train.servers) == {"student", "judge"} and train.tutor == pytest.approx(0.50)
        assert "eval_judge" in evaluating.servers and evaluating.tutor == pytest.approx(0.26)
        assert "| **total** | **80%** |" in train.markdown()
    # Sharing the student's server frees the judge's share for the tutor.
    shared = replace(cfg, judge={**cfg.judge, "share_student": True})
    assert plan(shared).tutor == pytest.approx(0.65)
    # A config that cannot fit fails up front, before any download.
    greedy = replace(cfg, student={**cfg.student, "gpu_memory_utilization": 0.5})
    with pytest.raises(ValueError, match="does not fit"):
        plan(greedy, evaluating=True)
