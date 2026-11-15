"""Resource-bounded PedagogicalRL-inspired LoRA training loop (issue #46, week 1).

Usage (one GPU with torch, transformers>=5, peft, datasets installed):
    python training/train.py --config training/configs/molab.toml --launch-servers

`--launch-servers` starts the student and judge as local vLLM servers (vLLM is installed in its
own venv). Without it, point TRAIN_STUDENT_* and TRAIN_JUDGE_* (BASE_URL, MODEL, API_KEY) at
existing OpenAI-compatible endpoints. The marimo notebook `tutor_rl_molab.py` runs the same code.

Loop per step (mirrors PedagogicalRL's classroom, scaled down), see tutor_rl/grpo.py:
    1. Sample `problems_per_step` problems from the prepared train set.
    2. For each, roll out `episodes_per_problem` dialogues with `episode.run_episodes`
       (tutor = current LoRA policy, all dialogues batched in lockstep).
    3. Score each with `rewards.score_episodes`; compute `rewards.group_advantages` per problem.
    4. GRPO update on the tutor's own tokens only (student/system tokens masked out),
       with KL penalty `beta` against the base model.
    5. Every `save_steps`, save the LoRA adapter, the config, and the data manifest together.
"""

import argparse
import json
from pathlib import Path

from tutor_rl.config import load_config
from tutor_rl.data import prepare_data, read_problems
from tutor_rl.openai_chat import model_from_section


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", help="defaults to [train].output_dir")
    parser.add_argument("--launch-servers", action="store_true")
    parser.add_argument("--vllm-env", default=str(Path.home() / ".cache" / "tutor-rl-vllm"))
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or cfg.output_dir)
    data_dir = run_dir / "data"
    if not (data_dir / "manifest.json").exists():
        prepare_data(cfg, data_dir)

    servers = []
    if args.launch_servers:
        from tutor_rl.vllm_server import ensure_vllm_env, start_student_and_judge

        servers = start_student_and_judge(
            cfg, ensure_vllm_env(Path(args.vllm_env)), run_dir / "logs"
        )
    try:
        from tutor_rl.grpo import GRPOTrainer
        from tutor_rl.policy import load_tutor

        trainer = GRPOTrainer(
            cfg,
            load_tutor(cfg.tutor),
            model_from_section(cfg.student),
            model_from_section(cfg.judge),
            read_problems(data_dir / "train.jsonl"),
            run_dir,
        )
        if args.resume:
            print(f"Resumed at step {trainer.resume()}")
        final = trainer.train(on_step=lambda m: print(json.dumps(m)))
        print(f"Final adapter: {final / 'adapter'}")
    finally:
        for s in servers:
            s.stop()


if __name__ == "__main__":
    main()
