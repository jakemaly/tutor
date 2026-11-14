"""Resource-bounded PedagogicalRL-inspired QLoRA training loop (issue #46, week 1).

Usage (on a GPU machine with unsloth, trl, vllm installed; student/judge endpoint in env):
    python training/train.py --config training/configs/small.toml

Loop per step (mirrors PedagogicalRL's classroom, scaled down):
    1. Sample `problems_per_step` problems from the prepared train set.
    2. For each, roll out `episodes_per_problem` dialogues with `episode.run_episode`
       (tutor = current QLoRA policy via Unsloth's vLLM fast generation).
    3. Score each with `rewards.score_episode`; compute `rewards.group_advantages` per problem.
    4. GRPO update on the tutor's own tokens only (student/system tokens masked out),
       with KL penalty `beta` against the base model.
    5. Every `save_steps`, save the LoRA adapter, the config, and the data manifest together.
"""

import argparse

from tutor_rl.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    # TODO(#46): implement steps 1-5. Reuse the Unsloth model/LoRA setup from the GRPO onboarding
    # notebook (notebooks/ml_onboarding/JaylenSze.py on branch ml-onboarding-JaylenSze).
    # Multi-turn GRPO is not what TRL's GRPOTrainer does out of the box (it scores one completion
    # per prompt), so either write the loss directly or subclass it, as the authors did in
    # eth-lre/PedagogicalRL src/grpo/trainer.py.
    raise NotImplementedError(f"Training for {cfg.run_name} is not implemented yet.")


if __name__ == "__main__":
    main()
