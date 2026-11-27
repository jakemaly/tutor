"""Versioned problem sets -> `Problem` JSONL plus a manifest (the "data version").

The paper's problems (section 5.2, Table 1) are Big-Math-RL-Verified rows with one numerical
answer and a Llama-3.1-8B-Instruct solve rate between 1% and 60% (of 64 samples), split into
10,000 train and 500 test problems. That exact split is published by the paper's first author
as `rd211/Big-Math-RL-Verified-Filtered` (train/test per-source counts equal Table 1's).
"""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from tutor_rl.config import RunConfig
from tutor_rl.episode import Problem
from tutor_rl.rewards import normalize_answer


def bigmath_problem(row: dict, split: str, index: int) -> Problem:
    return Problem(
        id=f"bigmath-{split}-{index}",
        question=row["problem"].strip(),
        answer=row["answer"].strip(),
        source=row.get("source", ""),
        solve_rate=row.get("llama8b_solve_rate"),
    )


def gsm8k_problem(row: dict, split: str, index: int) -> Problem:
    """GSM8K stores the worked solution; the final answer is the text after '####'."""
    solution = row["answer"]
    if "####" not in solution:
        raise ValueError(f"GSM8K {split}[{index}] has no '####' answer")
    return Problem(
        id=f"gsm8k-{split}-{index}",
        question=row["question"].strip(),
        answer=normalize_answer(solution.split("####")[-1]),
        source="gsm8k",
    )


FORMATS = {"bigmath": bigmath_problem, "gsm8k": gsm8k_problem}


def write_problems(path: Path, problems: Iterable[Problem]) -> str:
    """Write one problem per line and return the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(vars(p), ensure_ascii=False) + "\n" for p in problems)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def read_problems(path: Path) -> list[Problem]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Problem(**json.loads(line)) for line in lines if line.strip()]


def prepare_data(cfg: RunConfig, data_dir: Path) -> dict:
    """Download the pinned dataset revision, write train/eval JSONL and manifest.json."""
    from datasets import load_dataset  # only needed on the machine that prepares data

    d = cfg.data
    to_problem = FORMATS[d.get("format", "bigmath")]
    manifest = {
        "dataset": d["dataset"],
        "config": d.get("dataset_config"),
        "revision": d["revision"],
        "format": d.get("format", "bigmath"),
        "files": {},
    }
    for name in ("train", "eval"):
        split, limit = d[f"{name}_split"], d[f"{name}_limit"]
        rows = load_dataset(
            d["dataset"], d.get("dataset_config"), split=split, revision=d["revision"]
        )
        problems = [to_problem(rows[i], split, i) for i in range(min(limit, len(rows)))]
        sha = write_problems(data_dir / f"{name}.jsonl", problems)
        sources: dict[str, int] = {}
        for p in problems:
            sources[p.source] = sources.get(p.source, 0) + 1
        manifest["files"][f"{name}.jsonl"] = {
            "split": split,
            "limit": limit,
            "count": len(problems),
            "sources": dict(sorted(sources.items())),
            "sha256": sha,
        }
    # One id for the whole data version: hash of the per-file hashes.
    joined = "".join(f["sha256"] for f in manifest["files"].values())
    manifest["data_version"] = hashlib.sha256(joined.encode()).hexdigest()[:16]
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
