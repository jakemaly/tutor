"""Base vs. trained tutor on the paper's held-out Big-Math test set (paper section 5.2, Table 2).

Every policy gets the same problems, openers, sampling seeds, student, and judges. Metrics:

- delta_solve_rate: post-dialog minus pre-dialog student solve rate, in percentage points.
  The pre-dialog rate is the student's chain-of-thought accuracy (Figure 9 prompt, K samples),
  measured once and shared by every policy.
- leak_rate: share of dialogues the leakage judge (Figure 13) rejects. The paper uses a judge
  from another family at test time (Gemma3-27B); configure it as [eval_judge]. Without one,
  the training judge is used and the result says so.
- reward and its parts with the training judges, as during training.
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
from tutor_rl.rewards import (
    parse_rejection,
    run_judges,
    score_dict,
    score_episodes,
    solve_rate,
    summarize,
)

METRICS = (
    "delta_solve_rate",
    "leak_rate",
    "post_solve_rate",
    "pre_solve_rate",
    "reward",
    "accept_rate",
    "helpfulness_reject_rate",
    "mean_turns",
    "ended_by_tutor_rate",
)

# Paper Table 2 (in-domain Big-Math test set, Gemma3-27B leak judge), for reference.
PAPER_REFERENCE = {
    "Qwen2.5-7B-Instruct": (11.3, 29.3),
    "Qwen2.5-14B-Instruct": (29.3, 41.9),
    "Qwen2.5-72B-Instruct": (38.7, 61.0),
    "GPT-4o-2024-11-20": (33.1, 35.2),
    "LearnLM 2.0 Flash": (4.3, 0.9),
    "SocraticLM": (15.9, 40.4),
    "Qwen2.5-7B-RL λ=0.0": (36.2, 89.5),
    "Qwen2.5-7B-RL λ=0.25": (29.3, 32.0),
    "Qwen2.5-7B-RL λ=0.5": (30.9, 25.1),
    "Qwen2.5-7B-RL λ=0.75": (25.3, 10.6),
    "Qwen2.5-7B-RL λ=1.0": (24.7, 18.4),
    "Qwen2.5-7B-RL λ=1.25": (29.1, 15.1),
    "Qwen2.5-7B-RL λ=1.5": (21.2, 5.4),
    "Qwen2.5-7B-RL-hard λ=1.0": (12.6, 5.3),
}


def _set_seed(model, seed: int | None) -> None:
    """Fixed sampling seeds for every model that supports them (HF policy, vLLM endpoints)."""
    if hasattr(model, "seed"):
        model.seed = seed
    if hasattr(model, "_calls"):
        model._calls = 0


def pre_dialog_solve_rates(
    problems: list[Problem], student: ChatModel, k: int, out_dir: Path
) -> list[float]:
    """Student's solo chain-of-thought accuracy per problem; cached in pre_dialog.jsonl."""
    path = out_dir / "pre_dialog.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if [r["id"] for r in rows] == [p.id for p in problems] and rows[0].get("k") == k:
            return [r["solve_rate"] for r in rows]
    attempts = generate_many(student, [pre_dialog_prompt(p) for p in problems], n=k)
    rates = [solve_rate(a, p.answer) for p, a in zip(problems, attempts, strict=True)]
    with open(path, "w", encoding="utf-8") as f:
        for p, a, r in zip(problems, attempts, rates, strict=True):
            row = {"id": p.id, "k": k, "solve_rate": r, "dataset_solve_rate": p.solve_rate}
            f.write(json.dumps({**row, "attempts": a}, ensure_ascii=False) + "\n")
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
    eval_judge: ChatModel | None = None,
    final_student: ChatModel | None = None,
    on_progress=None,  # optional callback(str) for notebook status lines
) -> dict:
    """Run every policy on the same scenarios; write results.json, results.md, episodes JSONL."""
    e = cfg.eval
    problems = problems[: e.get("problems", len(problems))]
    per_problem = e.get("episodes_per_problem", 1)
    scenarios = [p for p in problems for _ in range(per_problem)]
    seed = e.get("seed", cfg.seed)
    openers = pick_initiators(len(problems), cfg.episode, random.Random(seed))
    initiators = [who for who in openers for _ in range(per_problem)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leak_judge = eval_judge or judge
    solver = final_student or student
    say = on_progress or (lambda _msg: None)

    models = (*policies.values(), student, judge, leak_judge, solver)
    saved_seeds = {id(m): getattr(m, "seed", None) for m in models}
    # A smaller tutor batch while the eval judge also occupies the GPU (local HF policies only).
    saved_batch = {}
    if e.get("generation_batch_size"):
        for p in policies.values():
            if isinstance(getattr(p, "cfg", None), dict):
                saved_batch.setdefault(id(p), (p, p.cfg.get("generation_batch_size")))
                p.cfg = {**p.cfg, "generation_batch_size": e["generation_batch_size"]}
    try:
        return _evaluate(
            cfg, policies, problems, scenarios, initiators, student, judge, leak_judge, solver,
            out_dir, before_each, models, seed, eval_judge, say,
        )  # fmt: skip
    finally:
        for model in models:  # back to unseeded sampling
            if hasattr(model, "seed"):
                model.seed = saved_seeds[id(model)]
        for p, size in saved_batch.values():
            p.cfg = {**p.cfg, "generation_batch_size": size}


def _evaluate(
    cfg, policies, problems, scenarios, initiators, student, judge, leak_judge, solver,
    out_dir, before_each, models, seed, eval_judge, say,
) -> dict:  # fmt: skip
    e = cfg.eval
    for model in models:
        _set_seed(model, seed)
    say("Measuring the student's pre-dialog solve rate…")
    pre = pre_dialog_solve_rates(problems, solver, cfg.episode.student_final_attempts, out_dir)
    pre_by_id = dict(zip([p.id for p in problems], pre, strict=True))

    results = {}
    for name, policy in policies.items():
        if before_each:
            before_each(name, policy)
        for model in models:
            _set_seed(model, seed)
        say(f"{name}: running {len(scenarios)} dialogues…")
        episodes = run_episodes(
            scenarios, policy, student, cfg.episode, initiators=initiators, final_student=solver
        )
        say(f"{name}: judging…")
        scores = score_episodes(episodes, judge, cfg.reward)
        leak_out = run_judges(episodes, leak_judge, ["leakage"], e.get("leak_judge_samples", 1))
        leaked = [any(parse_rejection(o) for o in row["leakage"]) for row in leak_out]
        summary = summarize(episodes, scores)
        post = summary["solve_rate"]
        pre_mean = statistics.fmean(pre_by_id[ep.problem.id] for ep in episodes)
        results[name] = {
            **summary,
            "post_solve_rate": post,
            "pre_solve_rate": pre_mean,
            "delta_solve_rate": 100 * (post - pre_mean),
            "leak_rate": 100 * statistics.fmean(leaked),
            "train_judge_leak_rate": 100 * summary["leak_rate"],
        }
        with open(out_dir / f"episodes-{name}.jsonl", "w", encoding="utf-8") as f:
            for ep, sc, lk, lo in zip(episodes, scores, leaked, leak_out, strict=True):
                row = {**ep.to_dict(), **score_dict(sc), "eval_leaked": lk}
                row["eval_judge_outputs"] = lo["leakage"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "run_name": cfg.run_name,
        "seed": seed,
        "problems": len(problems),
        "scenarios": len(scenarios),
        "leak_judge": cfg.eval_judge.get("model") if eval_judge else cfg.judge.get("model"),
        "results": results,
        "paper_reference": PAPER_REFERENCE,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    (out_dir / "results.md").write_text(results_markdown(report), encoding="utf-8")
    return report


def _fmt(key: str, value) -> str:
    if value is None:
        return "–"
    if key in ("delta_solve_rate", "leak_rate"):
        return f"{value:.1f}"
    return f"{value:.3f}"


def results_markdown(report: dict) -> str:
    lines = [
        f"# Evaluation: {report['run_name']}",
        "",
        f"{report['scenarios']} dialogues on {report['problems']} held-out Big-Math problems, "
        f"seed {report['seed']}. Leak judge: `{report['leak_judge']}`. Δ solve rate and leak "
        "rate are percentages, as in the paper's Table 2.",
        "",
        "| Policy | " + " | ".join(METRICS) + " |",
        "| --- |" + " ---: |" * len(METRICS),
    ]
    for name, r in report["results"].items():
        lines.append(f"| {name} | " + " | ".join(_fmt(m, r.get(m)) for m in METRICS) + " |")
    lines += [
        "",
        "Paper Table 2 for reference (Qwen2.5 tutors, Gemma3-27B leak judge):",
        "",
        "| Model | Δ solve rate (%) | Leaked solution (%) |",
        "| --- | ---: | ---: |",
    ]
    for name, (delta, leak) in report.get("paper_reference", {}).items():
        lines.append(f"| {name} | {delta:.1f} | {leak:.1f} |")
    return "\n".join(lines) + "\n"
