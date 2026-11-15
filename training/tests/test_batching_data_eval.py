"""Batched rollouts, the OpenAI-compatible client, data files, and evaluation (no GPU needed)."""

import json
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from tutor_rl.config import load_config
from tutor_rl.data import gsm8k_problem, read_problems, write_problems
from tutor_rl.episode import Message, Problem, run_episodes
from tutor_rl.evaluation import evaluate
from tutor_rl.openai_chat import OpenAIChatModel
from tutor_rl.prompts import END_TOKEN

CONFIGS = Path(__file__).parents[1] / "configs"


class Echo:
    """Batched fake: replies with a function of the conversation; counts batch calls."""

    def __init__(self, reply) -> None:
        self.reply = reply
        self.batches: list[int] = []

    def generate(self, messages: Sequence[Message], *, n: int = 1) -> list[str]:
        return self.generate_batch([messages], n=n)[0]

    def generate_batch(self, conversations, *, n: int = 1) -> list[list[str]]:
        self.batches.append(len(conversations))
        return [[self.reply(c)] * n for c in conversations]


def _problem(i: int) -> Problem:
    return Problem(id=f"p{i}", question=f"What is {i} + {i}?", answer=str(2 * i))


def test_lockstep_rollout_batches_calls_and_records_tutor_steps() -> None:
    # The tutor ends the dialogue on its second turn only for problem p0.
    def tutor_reply(conv):
        tutor_turns = sum(m.role == "assistant" for m in conv)
        if "0 + 0" in conv[0].content and tutor_turns == 1:
            return END_TOKEN
        return "What do you get if you add them?"

    tutor = Echo(tutor_reply)
    student = Echo(lambda conv: "\\boxed{0}" if "over" in conv[-1].content else "hmm")
    cfg = load_config(CONFIGS / "molab.toml").episode
    episodes = run_episodes([_problem(0), _problem(1)], tutor, student, cfg)

    # Student, tutor, student, tutor(p0 ends), student(p1), ... then one final batch.
    assert tutor.batches == [2, 2, 1, 1]
    assert student.batches[:2] == [2, 2] and student.batches[-1] == 2
    p0, p1 = episodes
    assert p0.ended_by_tutor and len(p0.turns) == 3
    assert [s.context_turns for s in p0.tutor_steps] == [1, 3]
    assert p0.tutor_steps[-1].output == END_TOKEN
    assert not p1.ended_by_tutor and len(p1.turns) == cfg.max_turns
    # The prompt the trainer rebuilds is the prompt the tutor saw.
    assert p1.tutor_view(p1.tutor_steps[1].context_turns)[-1].content == "hmm"
    assert p0.final_attempts == ["\\boxed{0}"] * cfg.student_final_attempts


class _ChatHandler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _ChatHandler.seen.append(body)
        last = body["messages"][-1]["content"]
        choices = [
            {"index": i, "message": {"role": "assistant", "content": f"{last}#{i}"}}
            for i in reversed(range(body["n"]))
        ]
        payload = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_openai_client_batches_with_seeds(monkeypatch) -> None:
    server = HTTPServer(("127.0.0.1", 0), _ChatHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("T_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        monkeypatch.setenv("T_MODEL", "student")
        model = OpenAIChatModel.from_env("T", temperature=0.5, max_tokens=7)
        model.seed = 10
        out = model.generate_batch([[Message("user", "a")], [Message("user", "b")]], n=2)
        assert out == [["a#0", "a#1"], ["b#0", "b#1"]]
        seen = sorted(_ChatHandler.seen, key=lambda b: b["seed"])
        assert [b["seed"] for b in seen] == [10, 11]
        assert seen[0]["model"] == "student" and seen[0]["max_tokens"] == 7
    finally:
        server.shutdown()


def test_gsm8k_rows_and_problem_files(tmp_path) -> None:
    row = {"question": " Tom has 3 boxes... ", "answer": "3*400=1200\n#### 1,200"}
    problem = gsm8k_problem(row, "train", 5)
    assert problem == Problem("gsm8k-train-5", "Tom has 3 boxes...", "1200")
    sha = write_problems(tmp_path / "train.jsonl", [problem])
    assert len(sha) == 64 and read_problems(tmp_path / "train.jsonl") == [problem]


def test_evaluate_compares_policies_on_same_scenarios(tmp_path) -> None:
    cfg = load_config(CONFIGS / "smoke.toml")
    leaky = Echo(lambda conv: f"The answer is \\boxed{{2}} {END_TOKEN}")
    socratic = Echo(lambda conv: f"What is 1 + 1? {END_TOKEN}")
    student = Echo(lambda conv: "\\boxed{2}")
    judge = Echo(
        lambda conv: (
            '{"decision": "REJECT"}' if "answer is" in conv[0].content else '{"decision": "OK"}'
        )
    )
    report = evaluate(
        cfg, {"base": leaky, "trained": socratic}, [_problem(1)] * 3, student, judge, tmp_path
    )
    assert report["scenarios"] == cfg.eval["problems"] * cfg.eval["episodes_per_problem"]
    assert report["results"]["base"]["leak_rate"] == 1.0
    assert report["results"]["trained"]["reward"] == 1.0
    assert "| trained |" in (tmp_path / "results.md").read_text()
    assert (tmp_path / "episodes-base.jsonl").read_text().count("\n") == report["scenarios"]
