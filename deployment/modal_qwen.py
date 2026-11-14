"""Configuration for serving the tutor's model on Modal."""

import modal

app = modal.App("socratic-tutor-qwen")

MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
MODEL_REVISION = "42c29f9ead88634cbc48378a71d1f55f7c76ad70"

TOKENIZER_REPO = "Qwen/Qwen3.5-9B"
TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SERVED_MODEL_NAME = "socratic-tutor"

GGUF_PLUGIN_REVISION = "e454affe0ca23c9dc8754ec3033270cb30622ed5"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git", "build-essential")
    .pip_install(
        "vllm==0.32.0",
        "ninja",
        "wheel",
        "setuptools>=77.0.3,<81",
    )
    .env({
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "MAX_JOBS": "2",
    })
    .pip_install(
        "vllm-gguf-plugin @ "
        "git+https://github.com/vllm-project/vllm-gguf-plugin.git"
        f"@{GGUF_PLUGIN_REVISION}",
        extra_options="--no-build-isolation",
    )
    .run_commands(
        # Compatibility fix for vLLM 0.32.0 (GGUF plugin PR #148).
        r"sed -i -E 's/^from vllm\.transformers_utils\.configs\.qwen3_5(_moe)? import /from transformers import /' "
        "/usr/local/lib/python3.12/site-packages/vllm_gguf_plugin/weights_adapter/qwen3_5.py",
        "python -c 'from vllm_gguf_plugin import register'",
    )
)

model_cache = modal.Volume.from_name(
    "socratic-tutor-model-cache",
    create_if_missing=True,
)


@app.function(
    image=image.env({"HF_HOME": "/cache/huggingface"}),
    gpu=["L4", "L40S"],
    cpu=2,
    memory=16384,
    volumes={"/cache": model_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=600,
    startup_timeout=900,
)
@modal.web_server(
    port=8000,
    startup_timeout=900,
    requires_proxy_auth=True,
)
def serve():
    """Start the model server inside the remote container."""
    import subprocess

    from huggingface_hub import snapshot_download

    config_dir = snapshot_download(
        repo_id=TOKENIZER_REPO,
        revision=TOKENIZER_REVISION,
        allow_patterns=["*.json", "*.jinja", "*.txt"],
        local_dir="/cache/qwen-config",
    )

    command = [
        "vllm",
        "serve",
        f"{MODEL_REPO}:Q4_K_M",
        "--revision", MODEL_REVISION,
        "--tokenizer", TOKENIZER_REPO,
        "--tokenizer-revision", TOKENIZER_REVISION,
        "--hf-config-path", config_dir,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--download-dir", "/cache/huggingface/hub",
        # Use the longest context that fits this GPU; /v1/models reports the chosen size.
        "--max-model-len", "auto",
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.85",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--enforce-eager",
    ]

    subprocess.Popen(command)