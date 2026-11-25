"""CPU sensitivity check: ask for a short boxed answer on the saved pilot dialogues."""

import argparse
import hashlib
import json
from pathlib import Path

from compare import HERE, probe
from cpu_pilot import pilot_models
from tutor_rl.episode import Episode, Problem, Turn
from tutor_rl.rewards import last_boxed, solve_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, _, solver, _ = pilot_models(args.gguf)
    targets = json.loads((HERE / "targets.json").read_text())
    raw = args.report.read_bytes()
    rows = []
    cold = {}
    for row in json.loads(raw)["rows"]:
        key = row["source_id"]
        t = targets[key]
        target = Problem(key + "-transfer", t["question"], t["answer"])
        k = row["shared_cold_samples"]
        solver.seed = row["seed"]
        ep = Episode(
            Problem(**row["episode"]["problem"]),
            turns=[Turn(**turn) for turn in row["episode"]["turns"]],
        )
        if key not in cold:
            cold[key] = solver.generate(probe(target, answer_only=True), n=k)
        answers = solver.generate(probe(target, ep, answer_only=True), n=k)
        rows.append(
            dict(
                source_id=key,
                arm=row["arm"],
                cold_attempts=cold[key],
                transfer_attempts=answers,
                cold=solve_rate(cold[key], target.answer),
                transfer=solve_rate(answers, target.answer),
                boxed=sum(last_boxed(a) is not None for a in answers),
            )
        )
    args.output.write_text(
        json.dumps(dict(source_report_sha256=hashlib.sha256(raw).hexdigest(), rows=rows), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
