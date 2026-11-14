"""Build the versioned problem set the trainer and evaluator read (issue #46, week 1).

Usage (on the training machine, with `datasets` installed):
    python training/prepare_data.py --config training/configs/small.toml

Writes outputs/<run>/data/{train,eval}.jsonl with one `Problem` per line plus a manifest that
records the dataset name, pinned revision, split, limits, and a content hash (the "data version").
"""

import argparse

from tutor_rl.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    # TODO(#46): load cfg.data["dataset"] at cfg.data["revision"] with `datasets`, map GSM8K
    # rows to Problem(id, question, answer=text after "####"), take the first *_limit rows,
    # write JSONL, and write manifest.json with a sha256 of each file.
    raise NotImplementedError(f"Data preparation for {cfg.run_name} is not implemented yet.")


if __name__ == "__main__":
    main()
