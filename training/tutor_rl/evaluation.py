"""Untrained vs. trained tutor on the same fixed scenarios (same problems, openers, seeds,
student and judge).

Per tutor:
- leak_rate: % of dialogues the leakage judge rejects (the tutor gave the answer away).
- post_solve_rate: % of the student's solutions after the dialogue that are correct.
- pre_solve_rate: % correct when the student solves alone, before any dialogue (same for both).
- delta_solve_rate: post minus pre, in percentage points: what the tutoring added.
"""

import json
import random
import statistics
from pathlib import Path

from tutor_rl.config import RunConfig
from tutor_rl.episode import (
    ChatModel,
    Problem,
    generate_many,
    pick_initiators,
    pre_dialog_prompt,
    run_episodes,
)
from tutor_rl.rewards import score_dict, score_episodes, solve_rate, summarize

METRICS = (
    ("leak_rate", "Answer leaked (%)"),
    ("post_solve_rate", "Student solved after dialogue (%)"),
    ("delta_solve_rate", "Gain over student alone (pp)"),
    ("accept_rate", "Judges accept (%)"),
    ("reward", "Mean reward"),
    ("mean_turns", "Messages per dialogue"),
)


def _set_seed(model, seed: int | None) -> None:
    """Fixed sampling seeds for every model that supports them (HF policy, vLLM endpoints)."""
    if hasattr(model, "seed"):
        model.seed = seed
    if hasattr(model, "_calls"):
        model._calls = 0


def pre_dialog_solve_rates(
    problems: list[Problem], student: ChatModel, k: int, out_dir: Path
) -> list[float]:
    """The student's solo solve rate per problem; cached in pre_dialog.jsonl."""
    path = out_dir / "pre_dialog.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if [r["id"] for r in rows] == [p.id for p in problems] and rows[0].get("k") == k:
            return [r["solve_rate"] for r in rows]
    attempts = generate_many(student, [pre_dialog_prompt(p) for p in problems], n=k)
    rates = [solve_rate(a, p.answer) for p, a in zip(problems, attempts, strict=True)]
    with open(path, "w", encoding="utf-8") as f:
        for p, a, r in zip(problems, attempts, rates, strict=True):
            f.write(json.dumps({"id": p.id, "k": k, "solve_rate": r, "attempts": a}) + "\n")
    return rates


def evaluate(
    cfg: RunConfig,
    policies: dict[str, ChatModel],
    problems: list[Problem],
    student: ChatModel,
    judge: ChatModel,
    out_dir: Path,
    before_each=None,  # optional callback(name, policy), e.g. to switch an adapter on or off
    *,
    final_student: ChatModel | None = None,
) -> dict:
    """Run every policy on the same scenarios; write results.json, results.md, episodes JSONL."""
    e = cfg.eval
    problems = problems[: e.get("problems", len(problems))]
    per_problem = e.get("episodes_per_problem", 1)
    scenarios = [p for p in problems for _ in range(per_problem)]
    seed = e.get("seed", cfg.seed)
    openers = pick_initiators(len(problems), cfg.episode, random.Random(seed))
    initiators = [who for who in openers for _ in range(per_problem)]
    solver = final_student or student
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = (*policies.values(), student, judge, solver)
    saved_seeds = {id(m): getattr(m, "seed", None) for m in models}
    try:
        for model in models:
            _set_seed(model, seed)
        pre = dict(
            zip(
                [p.id for p in problems],
                pre_dialog_solve_rates(
                    problems, solver, cfg.episode.student_final_attempts, out_dir
                ),
                strict=True,
            )
        )
        results = {}
        for name, policy in policies.items():
            if before_each:
                before_each(name, policy)
            for model in models:
                _set_seed(model, seed)
            episodes = run_episodes(
                scenarios, policy, student, cfg.episode, initiators=initiators, final_student=solver
            )
            scores = score_episodes(episodes, judge, cfg.reward)
            s = summarize(episodes, scores)
            pre_mean = statistics.fmean(pre[ep.problem.id] for ep in episodes)
            results[name] = {
                **s,
                "leak_rate": 100 * s["leak_rate"],
                "accept_rate": 100 * s["accept_rate"],
                "post_solve_rate": 100 * s["solve_rate"],
                "pre_solve_rate": 100 * pre_mean,
                "delta_solve_rate": 100 * (s["solve_rate"] - pre_mean),
            }
            with open(out_dir / f"episodes-{name}.jsonl", "w", encoding="utf-8") as f:
                for ep, sc in zip(episodes, scores, strict=True):
                    f.write(json.dumps({**ep.to_dict(), **score_dict(sc)}, ensure_ascii=False))
                    f.write("\n")
    finally:
        for model in models:  # back to unseeded sampling
            if hasattr(model, "seed"):
                model.seed = saved_seeds[id(model)]

    report = {
        "run_name": cfg.run_name,
        "seed": seed,
        "scenarios": len(scenarios),
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    (out_dir / "results.md").write_text(results_markdown(report), encoding="utf-8")
    return report


def results_markdown(report: dict) -> str:
    names = list(report["results"])
    pre = next(iter(report["results"].values()))["pre_solve_rate"]
    lines = [
        f"{report['scenarios']} dialogues per tutor (same problems, openers, seeds, student and "
        f"judge). The student alone solves {pre:.1f}% of these problems.",
        "",
        "| | " + " | ".join(names) + " |",
        "| --- |" + " ---: |" * len(names),
    ]
    for key, label in METRICS:
        values = [report["results"][n].get(key) for n in names]
        fmt = ".2f" if key == "reward" else ".1f"
        lines.append(f"| {label} | " + " | ".join(f"{v:{fmt}}" for v in values) + " |")
    return "\n".join(lines) + "\n"
