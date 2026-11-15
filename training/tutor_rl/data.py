"""Versioned problem sets: GSM8K rows -> `Problem` JSONL plus a manifest (the "data version")."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from tutor_rl.config import RunConfig
from tutor_rl.episode import Problem
from tutor_rl.rewards import normalize_answer


def gsm8k_problem(row: dict, split: str, index: int) -> Problem:
    """GSM8K stores the worked solution; the final answer is the text after '####'."""
    solution = row["answer"]
    if "####" not in solution:
        raise ValueError(f"GSM8K {split}[{index}] has no '####' answer")
    return Problem(
        id=f"gsm8k-{split}-{index}",
        question=row["question"].strip(),
        answer=normalize_answer(solution.split("####")[-1]),
    )


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
    manifest = {
        "dataset": d["dataset"],
        "config": d.get("dataset_config"),
        "revision": d["revision"],
        "files": {},
    }
    for name in ("train", "eval"):
        split, limit = d[f"{name}_split"], d[f"{name}_limit"]
        rows = load_dataset(
            d["dataset"], d.get("dataset_config"), split=split, revision=d["revision"]
        )
        problems = [gsm8k_problem(rows[i], split, i) for i in range(min(limit, len(rows)))]
        sha = write_problems(data_dir / f"{name}.jsonl", problems)
        manifest["files"][f"{name}.jsonl"] = {
            "split": split,
            "limit": limit,
            "count": len(problems),
            "sha256": sha,
        }
    # One id for the whole data version: hash of the per-file hashes.
    joined = "".join(f["sha256"] for f in manifest["files"].values())
    manifest["data_version"] = hashlib.sha256(joined.encode()).hexdigest()[:16]
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
