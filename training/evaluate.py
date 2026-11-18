"""Compare the trained checkpoint with the base tutor on the same fixed scenarios (week 2).

Usage:
    python training/evaluate.py --config training/configs/molab.toml \
        --policy base=hf:base --policy trained=hf:outputs/tutor-rl-qwen35-9b/final/adapter

    # or, once both are served through the OpenAI-compatible vLLM path:
    python training/evaluate.py --config training/configs/molab.toml \
        --policy base=BASE_ENDPOINT_ENV_PREFIX --policy trained=TRAINED_ENDPOINT_ENV_PREFIX

`hf:base` / `hf:<adapter dir>` load the tutor locally (one base model, adapter switched on or
off). `<ENV_PREFIX>` reads <PREFIX>_BASE_URL / _MODEL / _API_KEY, so the app's default provider
(LLM_BASE_URL in .env) is never changed. Reports the paper's Table 2 metrics per policy (delta
solve rate, leaked solution %), plus reward parts, on the same problems, seeds, student, judges.
The leak judge is [eval_judge] (Gemma3-27B) when EVAL_JUDGE_BASE_URL/_MODEL are set, else the
training judge. Writes results.json, results.md and per-policy episodes under <run_dir>/eval/.
"""

import argparse
import os
from pathlib import Path

from tutor_rl.config import load_config
from tutor_rl.data import prepare_data, read_problems
from tutor_rl.evaluation import evaluate
from tutor_rl.openai_chat import OpenAIChatModel, model_from_section


def build_policies(cfg, specs: list[str]):
    """Parse NAME=hf:base | NAME=hf:<adapter> | NAME=<ENV_PREFIX>."""
    policies, hf_adapters, hf_policy = {}, {}, None
    for spec in specs:
        name, _, target = spec.partition("=")
        if target.startswith("hf:"):
            hf_adapters[name] = target[3:]
        else:
            policies[name] = OpenAIChatModel.from_env(
                target,
                temperature=cfg.tutor.get("temperature", 1.0),
                max_tokens=cfg.tutor.get("max_tokens_per_turn", 256),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
    adapters = {n: a for n, a in hf_adapters.items() if a != "base"}
    if len(set(adapters.values())) > 1:
        raise SystemExit("Compare at most one local adapter per run (plus hf:base).")
    if hf_adapters:
        from tutor_rl.gpu_budget import cap_tutor, plan
        from tutor_rl.policy import load_tutor

        # The eval judge may be served on this GPU too: budget for it.
        cap_tutor(
            plan(
                cfg,
                evaluating=bool(
                    os.environ.get(f"{cfg.eval_judge.get('env_prefix', 'EVAL_JUDGE')}_BASE_URL")
                ),
            )
        )
        adapter = next(iter(adapters.values()), None)
        hf_policy = load_tutor(cfg.tutor, adapter_path=adapter, trainable=False)
        policies.update({n: hf_policy for n in hf_adapters})

    def switch(name, policy):
        if policy is hf_policy:
            policy.use_adapter = hf_adapters[name] != "base"

    return policies, switch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=TARGET")
    parser.add_argument("--run-dir", help="defaults to [train].output_dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or cfg.output_dir)
    if not args.policy:
        parser.error("give at least one --policy")
    if not (run_dir / "data" / "eval.jsonl").exists():
        prepare_data(cfg, run_dir / "data")

    policies, switch = build_policies(cfg, args.policy)
    eval_judge = None
    if cfg.eval_judge and os.environ.get(f"{cfg.eval_judge['env_prefix']}_BASE_URL"):
        eval_judge = model_from_section(cfg.eval_judge)
    else:
        print("No eval judge endpoint set: the training judge also measures leaks.")
    report = evaluate(
        cfg,
        policies,
        read_problems(run_dir / "data" / "eval.jsonl"),
        model_from_section(cfg.student),
        model_from_section(cfg.judge),
        run_dir / "eval",
        before_each=switch,
        eval_judge=eval_judge,
        final_student=model_from_section(cfg.student, max_tokens_key="max_tokens_final"),
        on_progress=print,
    )
    print((run_dir / "eval" / "results.md").read_text())
    return report


if __name__ == "__main__":
    main()
