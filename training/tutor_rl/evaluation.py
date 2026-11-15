"""Base vs. trained tutor on the same fixed scenarios (same problems, seeds, student, judge)."""

import json
from pathlib import Path

from tutor_rl.config import RunConfig
from tutor_rl.episode import ChatModel, Problem, run_episodes
from tutor_rl.rewards import score_episodes, summarize

METRICS = ("reward", "solve_rate", "leak_rate", "mean_turns", "ended_by_tutor_rate")


def _set_seed(model, seed: int) -> None:
    """Fixed sampling seeds for every model that supports them (HF policy, vLLM endpoints)."""
    if hasattr(model, "seed"):
        model.seed = seed
    if hasattr(model, "_calls"):
        model._calls = 0


def evaluate(
    cfg: RunConfig,
    policies: dict[str, ChatModel],
    problems: list[Problem],
    student: ChatModel,
    judge: ChatModel,
    out_dir: Path,
    before_each=None,  # optional callback(name, policy), e.g. to switch an adapter on or off
) -> dict:
    """Run every policy on the same scenarios; write results.json, results.md, episodes JSONL."""
    e = cfg.eval
    problems = problems[: e.get("problems", len(problems))]
    scenarios = [p for p in problems for _ in range(e.get("episodes_per_problem", 1))]
    seed = e.get("seed", cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    saved_seeds = {id(m): getattr(m, "seed", None) for m in (*policies.values(), student, judge)}
    for name, policy in policies.items():
        if before_each:
            before_each(name, policy)
        for model in (policy, student, judge):
            _set_seed(model, seed)
        episodes = run_episodes(scenarios, policy, student, cfg.episode)
        scores = score_episodes(episodes, judge, cfg.reward)
        results[name] = summarize(episodes, scores)
        with open(out_dir / f"episodes-{name}.jsonl", "w", encoding="utf-8") as f:
            for ep, sc in zip(episodes, scores, strict=True):
                f.write(json.dumps({**ep.to_dict(), **vars(sc)}, ensure_ascii=False) + "\n")
    for model in (*policies.values(), student, judge):  # back to unseeded sampling
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
    lines = [
        f"# Evaluation: {report['run_name']}",
        "",
        f"{report['scenarios']} scenarios, seed {report['seed']}. "
        "Same problems, student, and judge for every policy.",
        "",
        "| Policy | " + " | ".join(METRICS) + " |",
        "| --- |" + " ---: |" * len(METRICS),
    ]
    for name, r in report["results"].items():
        lines.append(f"| {name} | " + " | ".join(f"{r[m]:.3f}" for m in METRICS) + " |")
    return "\n".join(lines) + "\n"
