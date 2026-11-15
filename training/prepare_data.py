"""Build the versioned problem set the trainer and evaluator read (issue #46, week 1).

Usage (on the training machine, with `datasets` installed):
    python training/prepare_data.py --config training/configs/molab.toml

Writes <output_dir>/data/{train,eval}.jsonl with one `Problem` per line plus manifest.json,
which records the dataset name, pinned revision, splits, limits, and a sha256 per file. The
`data_version` in the manifest is copied into every checkpoint.
"""

import argparse
import json
from pathlib import Path

from tutor_rl.config import load_config
from tutor_rl.data import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", help="defaults to [train].output_dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or cfg.output_dir)
    manifest = prepare_data(cfg, run_dir / "data")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
