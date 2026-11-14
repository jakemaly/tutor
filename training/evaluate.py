"""Compare the trained checkpoint with the base tutor on the same fixed scenarios (week 2).

Usage:
    python training/evaluate.py --config training/configs/small.toml \
        --policy base=BASE_ENDPOINT_ENV_PREFIX --policy trained=TRAINED_ENDPOINT_ENV_PREFIX

Both policies are reached through OpenAI-compatible endpoints (the existing Modal vLLM path),
so the app's default provider (LLM_BASE_URL in .env) is never changed.
Reports, per policy: mean reward, student solve rate, leak rate, and mean dialogue length,
using the same eval problems, seeds, student, and judge for both.
"""

import argparse

from tutor_rl.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=ENV_PREFIX")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # TODO(#46): for each policy and eval problem, run_episode + score_episode with fixed seeds,
    # then write a results table (JSON + Markdown) under cfg.train["output_dir"].
    raise NotImplementedError(f"Evaluation for {cfg.run_name} is not implemented yet.")


if __name__ == "__main__":
    main()
