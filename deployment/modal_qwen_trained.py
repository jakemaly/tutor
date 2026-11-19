"""Serve the RL-trained tutor (issue #46) next to its base model on Modal, for comparison.

One vLLM server, two model names on the same OpenAI-compatible endpoint:
- `socratic-tutor-base`: Qwen3.5-9B in bf16, the model the adapter was trained on.
- `socratic-tutor-trained`: the same model with the trained LoRA adapter switched on.

This is a separate Modal app. The app's own deployment (`modal_qwen.py`, GGUF Q4_K_M) and its
`LLM_BASE_URL` are left alone. See docs/modal-serving.md for how to deploy and compare.
"""

import modal

app = modal.App("socratic-tutor-qwen-trained")

BASE_REPO = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # what training used
BASE_NAME = "socratic-tutor-base"

# The finished run on the Hugging Face Hub (private, so the Modal secret needs a read token).
RUN_REPO = "socratictutor/tutor-rl-qwen35-9b-bigmath"
RUN_REVISION = "main"  # pin a commit hash here to freeze the served checkpoint
ADAPTER_DIR = "final/adapter"
TRAINED_NAME = "socratic-tutor-trained"
LORA_RANK = 32  # [tutor] lora_rank in training/configs/molab.toml

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.32.0")
    .env({"HF_HOME": "/cache/huggingface"})
)

model_cache = modal.Volume.from_name("socratic-tutor-model-cache", create_if_missing=True)


def adapter_for_vllm(src: str, dst: str) -> str:
    """Copy a PEFT adapter, renaming its weights to the layout vLLM expects.

    Training loaded Qwen3.5 text-only (`model.layers.*`), but vLLM serves the full checkpoint,
    whose language model sits under `model.language_model.layers.*`. Without the rename vLLM
    would not match the adapter to any layer and would quietly serve the base model.
    """
    import shutil
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    src_dir, dst_dir = Path(src), Path(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    weights = load_file(str(src_dir / "adapter_model.safetensors"))
    old, new = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."
    renamed = {(new + k[len(old):] if k.startswith(old) else k): v for k, v in weights.items()}
    assert any(k.startswith(new) for k in renamed), "no LoRA weights for the language model"
    save_file(renamed, str(dst_dir / "adapter_model.safetensors"))
    shutil.copy(src_dir / "adapter_config.json", dst_dir / "adapter_config.json")
    return str(dst_dir)


@app.function(
    image=image,
    # bf16 weights take ~17 GB. Fallbacks when no L40S is free (Modal tries them in order).
    gpu=["L40S", "A100-40GB", "L4"],
    cpu=4,
    memory=32768,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface")],  # HF_TOKEN with read access to RUN_REPO
    min_containers=0,
    max_containers=1,
    scaledown_window=600,  # stays warm between the comparison's tutor turns
    timeout=1800,
    startup_timeout=1200,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=1200, requires_proxy_auth=True)
def serve():
    """Download the adapter, then start vLLM with both model names."""
    import json
    import subprocess

    from huggingface_hub import snapshot_download

    run_dir = snapshot_download(
        repo_id=RUN_REPO,
        revision=RUN_REVISION,
        allow_patterns=[f"{ADAPTER_DIR}/*"],
        local_dir="/cache/runs/" + RUN_REPO.split("/")[-1],
    )
    adapter = adapter_for_vllm(f"{run_dir}/{ADAPTER_DIR}", "/cache/runs/adapter-vllm")

    command = [
        "vllm",
        "serve",
        BASE_REPO,
        "--revision", BASE_REVISION,
        "--served-model-name", BASE_NAME,
        "--enable-lora",
        "--lora-modules", f"{TRAINED_NAME}={adapter}",
        "--max-lora-rank", str(LORA_RANK),
        "--max-loras", "1",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--download-dir", "/cache/huggingface/hub",
        "--max-model-len", "8192",
        "--max-num-seqs", "32",
        "--gpu-memory-utilization", "0.90",
        "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}),  # text only
        "--generation-config", "vllm",  # no sampling defaults from the model's files
        "--reasoning-parser", "qwen3",
        # The app lets the model call tools (tutor skills); Qwen3.5 writes them as <tool_call> XML.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--enforce-eager",
    ]

    subprocess.Popen(command)
