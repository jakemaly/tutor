"""Regression tests using PR #53's real rollout, scoring and masking boundaries."""

from types import SimpleNamespace

import pytest
from compare import ELICIT, Meter, compare, load_pairs, probe
from tutor_rl.config import EpisodeConfig, RewardConfig
from tutor_rl.episode import Episode, Message, Problem, Turn

SOURCE = Problem("source", "Original item", "7")
TARGET = Problem("target", "New item", "11")


class Recorder:
    def __init__(self, reply):
        self.calls = []
        self.reply = reply

    def generate(self, messages, *, n=1):
        self.calls.append((list(messages), n))
        return [self.reply(messages)] * n


def test_probe_masks_tutor_and_omits_answers():
    ep = Episode(SOURCE, turns=[Turn("tutor", "private hint"), Turn("student", "my method")])
    post = probe(TARGET, ep)[0].content
    assert "(hidden)" in post and "my method" in post
    assert all(secret not in post for secret in ("private hint", "Original item", "11"))
    assert "my method" not in probe(TARGET)[0].content


def test_paired_rollouts_and_counts():
    policy = Recorder(lambda m: "Can you explain?")
    student = Recorder(lambda m: "my method")
    solver = Recorder(lambda m: r"\boxed{11}" if "my method" in m[-1].content else r"\boxed{7}")
    judge = Recorder(lambda m: '{"decision":"REJECT"}')
    cfg = SimpleNamespace(episode=EpisodeConfig(2, 2, "student"), reward=RewardConfig())
    rows = compare([(SOURCE, TARGET)], cfg, policy, student, solver, judge, 42)
    assert [r["arm"] for r in rows] == ["baseline", "elicitation"]
    for r in rows:
        assert (r["source_solve"], r["cold"], r["transfer"], r["gain"]) == (1, 0, 1, 1)
        assert r["leakage"] and r["samples"] == 10 and r["shared_cold_samples"] == 2
        assert r["opener"] == "student" and r["seed"] == 42
    assert len(solver.calls) == 5  # one shared cold call, two source and two transfer calls
    base, alt = [m for m, _ in policy.calls]
    assert alt == [Message("system", base[0].content + "\n" + ELICIT), *base[1:]]
    assert all("New item" not in m.content for history, _ in policy.calls for m in history)


def test_incomplete_model_batch_fails():
    model = Recorder(lambda m: "")
    model.generate = lambda messages, *, n: []
    with pytest.raises(ValueError, match="Requested 2"):
        Meter(model).generate([], n=2)


def test_wrong_fixture_rejected(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text('{"id":"wrong","question":"q","answer":"0"}')
    with pytest.raises(ValueError, match="four test problems"):
        load_pairs(path)


def test_changed_source_with_matching_ids_rejected(tmp_path):
    import json

    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"bigmath-test-{i}", "question": "changed source", "answer": "1"})
            for i in range(4)
        )
    )
    with pytest.raises(ValueError, match="Source text differs"):
        load_pairs(path)


def test_short_retest_changes_only_answer_instruction():
    ep = Episode(SOURCE, turns=[Turn("tutor", "secret"), Turn("student", "my notes")])
    prompt = probe(TARGET, ep, answer_only=True)[0].content
    assert "secret" not in prompt and "my notes" in prompt
    assert TARGET.question in prompt and "only the final answer" in prompt
    assert TARGET.answer not in prompt and "step by step" not in prompt
