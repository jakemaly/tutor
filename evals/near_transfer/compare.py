"""Paired, inference-only comparison on PR #53's four Big-Math smoke scenarios."""

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

from tutor_rl.config import load_config
from tutor_rl.data import read_problems
from tutor_rl.episode import (
    Episode,
    Message,
    Problem,
    pick_initiators,
    pre_dialog_prompt,
    run_episodes,
)
from tutor_rl.openai_chat import model_from_section
from tutor_rl.rewards import score_episode, solve_rate

HERE = Path(__file__).resolve().parent
ELICIT = (
    "Ask the student to choose the next step and explain why it works. "
    "Before ending, ask them to describe in their own words a method they could use "
    "on a similar problem. Leave the key calculation to the student."
)


def load_pairs(path: Path) -> list[tuple[Problem, Problem]]:
    targets = json.loads((HERE / "targets.json").read_text())
    sources = read_problems(path)[: len(targets)]
    if [p.id for p in sources] != list(targets):
        raise ValueError("Expected the first four test problems from the pinned smoke fixture")
    pairs = []
    for p in sources:
        t = targets[p.id]
        if hashlib.sha256(p.question.encode()).hexdigest() != t["source_sha256"]:
            raise ValueError(f"Source text differs from pinned revision for {p.id}")
        pairs.append((p, Problem(p.id + "-transfer", t["question"], t["answer"])))
    return pairs


def probe(
    target: Problem, episode: Episode | None = None, *, answer_only: bool = False
) -> list[Message]:
    """Use the same test instruction with and without student-authored notes."""
    notes = (
        []
        if episode is None
        else [
            ["student", t.content] if t.speaker == "student" else ["tutor", "(hidden)"]
            for t in episode.turns
        ]
    )
    instruction = (
        target.question + r" Give only the final answer in \boxed{}."
        if answer_only
        else pre_dialog_prompt(target)[0].content
    )
    return [
        Message(
            "user",
            "Earlier dialogue (tutor messages hidden): "
            + json.dumps(notes, ensure_ascii=False)
            + "\n"
            + instruction,
        )
    ]


class Meter:
    """Count completed samples and apply the experimental instruction only to tutor calls."""

    def __init__(self, model, instruction=""):
        self.model = model
        self.instruction = instruction
        self.samples = 0

    def view(self, messages):
        if not self.instruction:
            return messages
        system, *history = messages
        if system.role != "system":
            raise ValueError("Expected the baseline tutor system prompt")
        return [Message("system", system.content + "\n" + self.instruction), *history]

    def generate(self, messages: Sequence[Message], *, n=1):
        replies = self.model.generate(self.view(messages), n=n)
        if len(replies) != n:
            raise ValueError(f"Requested {n} completions, received {len(replies)}")
        self.samples += n
        return replies

    def was_truncated(self, messages, text):
        check = getattr(self.model, "was_truncated", None)
        return bool(check and check(self.view(messages), text))


def compare(pairs, cfg, tutor, student, solver, judge, seed):
    k = cfg.episode.student_final_attempts
    openers = pick_initiators(len(pairs), cfg.episode, random.Random(seed))
    rows = []
    for index, ((source, target), opener) in enumerate(zip(pairs, openers, strict=True)):
        # Common random numbers within a scenario, including cold and masked probes.
        scenario_seed = seed + index * k
        for model in (tutor, student, solver, judge):
            model.seed = scenario_seed
        started = perf_counter()
        cold = Meter(solver).generate(probe(target), n=k)
        cold_seconds = perf_counter() - started
        for name, instruction in (("baseline", ""), ("elicitation", ELICIT)):
            for model in (tutor, student, solver, judge):
                model.seed = scenario_seed
            policy, pupil, final, assessor = (
                Meter(tutor, instruction),
                Meter(student),
                Meter(solver),
                Meter(judge),
            )
            started = perf_counter()
            (ep,) = run_episodes(
                [source],
                policy,
                pupil,
                cfg.episode,
                initiators=[opener],
                final_student=final,
            )
            score = score_episode(ep, assessor, cfg.reward)
            solver.seed = scenario_seed
            attempts = final.generate(probe(target, ep), n=k)
            post = solve_rate(attempts, target.answer)
            prior = solve_rate(cold, target.answer)
            rows.append(
                dict(
                    arm=name,
                    source_id=source.id,
                    seed=scenario_seed,
                    opener=opener,
                    cold=prior,
                    transfer=post,
                    gain=post - prior,
                    source_solve=score.solve_rate,
                    leakage=score.leaked,
                    seconds=perf_counter() - started,
                    samples=policy.samples + pupil.samples + final.samples + assessor.samples,
                    shared_cold_seconds=cold_seconds,
                    shared_cold_samples=k,
                    cold_attempts=cold,
                    transfer_attempts=attempts,
                    episode=ep.to_dict(),
                    score=asdict(score),
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Prepared smoke eval.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--gguf",
        type=Path,
        help="Run an optional small CPU pilot with one model in all roles",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    pairs = load_pairs(args.data)
    if args.gguf:
        from cpu_pilot import pilot_models

        models = pilot_models(args.gguf)
    else:
        from evaluate import build_policies

        policies, _ = build_policies(cfg, ["baseline=TUTOR_BASE"])
        models = (
            policies["baseline"],
            model_from_section(cfg.student),
            model_from_section(cfg.student, max_tokens_key="max_tokens_final"),
            model_from_section(cfg.judge),
        )
    seed = cfg.eval["seed"] if args.seed is None else args.seed
    tutor, student, solver, judge = models
    rows = compare(pairs, cfg, tutor, student, solver, judge, seed)
    summary = {}
    for arm in ("baseline", "elicitation"):
        subset = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            key: mean(r[key] for r in subset)
            for key in ("source_solve", "cold", "transfer", "gain", "leakage")
        }
        for key in ("samples", "seconds"):
            summary[arm][key] = sum(r[key] for r in subset)
    report = {
        "seed": seed,
        "models": [m.model for m in models],
        "summary": summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
