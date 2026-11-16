"""Reproduce paper Table 1: train/test composition by source with the student's solve rate.

Loads the pinned dataset revision from configs/molab.toml through tutor_rl.data (the same code
the trainer uses), counts problems per source, and averages the dataset's Llama-3.1-8B solve
rate per source. Writes artifacts/tables/table1_data_composition.{csv,md} and a comparison JSON
against the paper's printed values.

Usage (from training/): python replication/src/table1_data_composition.py
"""

import csv
import json
import statistics
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAINING))

from tutor_rl.config import load_config  # noqa: E402
from tutor_rl.data import FORMATS  # noqa: E402

# Paper Table 1 (samples, pre-dialog solve rate %), for the comparison only.
PAPER = {
    "train": {
        "big_math": (3360, 23.56), "cn_k12": (3324, 22.11), "math": (1264, 27.40),
        "aops_forum": (1263, 10.13), "omnimath": (374, 12.57), "openmath": (315, 38.18),
        "gsm8k": (100, 36.30),
    },
    "test": {
        "big_math": (177, 24.86), "cn_k12": (168, 22.34), "math": (57, 23.93),
        "aops_forum": (56, 10.07), "omnimath": (22, 15.41), "openmath": (13, 36.30),
        "gsm8k": (7, 32.14),
    },
}  # fmt: skip


def main() -> None:
    from datasets import load_dataset

    pin = json.loads((TRAINING / "replication" / "configs" / "table1.json").read_text())
    d = load_config(TRAINING / "configs" / "molab.toml").data
    for key in ("dataset", "revision", "format"):  # the trainer must use the same data
        assert d[key] == pin[key], f"configs/molab.toml [data].{key} != replication pin"
    out_dir = TRAINING / "replication" / "artifacts" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, comparison = [], {"dataset": d["dataset"], "revision": d["revision"], "splits": {}}
    for name, split in (("train", pin["train_split"]), ("test", pin["test_split"])):
        data = load_dataset(d["dataset"], split=split, revision=d["revision"])
        problems = [FORMATS[d["format"]](data[i], split, i) for i in range(len(data))]
        by_source: dict[str, list[float]] = {}
        for p in problems:
            by_source.setdefault(p.source, []).append(p.solve_rate)
        split_cmp = {"total": len(problems), "paper_total": sum(v[0] for v in PAPER[name].values())}
        max_rate_err = 0.0
        count_mismatches = []
        for source, (paper_n, paper_rate) in PAPER[name].items():
            rates = by_source.get(source, [])
            rate = 100 * statistics.fmean(rates) if rates else float("nan")
            rows.append([name, source, len(rates), round(rate, 2), paper_n, paper_rate])
            if len(rates) != paper_n:
                count_mismatches.append(source)
            max_rate_err = max(max_rate_err, abs(rate - paper_rate))
        split_cmp.update(
            count_mismatches=count_mismatches,
            max_solve_rate_abs_error_pp=round(max_rate_err, 3),
            extra_sources=sorted(set(by_source) - set(PAPER[name])),
        )
        comparison["splits"][name] = split_cmp

    header = [
        "split",
        "source",
        "samples",
        "solve_rate_pct",
        "paper_samples",
        "paper_solve_rate_pct",
    ]
    with open(out_dir / "table1_data_composition.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])
    md = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (out_dir / "table1_data_composition.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "table1_comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
