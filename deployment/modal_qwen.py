"""Configuration for serving the tutor's model on Modal."""

import os
import time

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


HF_HUB_DIR = "/cache/huggingface/hub"
MODEL_SNAPSHOT = (
    f"{HF_HUB_DIR}/models--unsloth--Qwen3.5-9B-GGUF/snapshots/{MODEL_REVISION}"
)
TOKENIZER_SNAPSHOT = (
    f"{HF_HUB_DIR}/models--Qwen--Qwen3.5-9B/snapshots/{TOKENIZER_REVISION}"
)
CONFIG_DIR = "/cache/qwen-config"
# vLLM compiles about 1,100 Triton kernel files while profiling Qwen3.5; keeping them on the
# Volume lets later cold containers load them instead of recompiling (about 60 s on L40S).
# Triton keys entries by kernel source, Triton version, and GPU target (L4 and L40S: sm_89).
TRITON_CACHE_DIR = "/cache/triton-cache"


def _cache_state() -> dict:
    """Report which model files and compiled-kernel caches exist before vLLM starts."""
    from pathlib import Path

    from cold_start_timing import directory_state, file_state

    return {
        "gguf": file_state(f"{MODEL_SNAPSHOT}/{MODEL_FILE}"),
        "mmproj": file_state(f"{MODEL_SNAPSHOT}/mmproj-BF16.gguf"),
        "tokenizer_snapshot": directory_state(TOKENIZER_SNAPSHOT),
        "config_dir": directory_state(CONFIG_DIR),
        "triton_cache": directory_state(TRITON_CACHE_DIR),
        "flashinfer_cache": directory_state(
            Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", Path.home())) / ".cache/flashinfer"
        ),
        "vllm_cache": directory_state(
            os.environ.get("VLLM_CACHE_ROOT", Path.home() / ".cache/vllm")
        ),
    }


def _forward_vllm_output(process, spawned: float) -> None:
    """Print each vLLM log line with its offset from spawn, preserving the original text."""
    for line in process.stdout:
        print(f"[vllm +{time.monotonic() - spawned:.3f}s] {line}", end="", flush=True)


def _local_vllm(
    path: str,
    method: str = "GET",
    body: dict | None = None,
    timeout=2,
    headers: dict | None = None,
) -> str | None:
    """Call the in-container vLLM server; return the body on HTTP 200, otherwise None."""
    import json
    import urllib.request

    request = urllib.request.Request(
        f"http://127.0.0.1:8000{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            **({"Content-Type": "application/json"} if body is not None else {}),
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode() if response.status == 200 else None
    except OSError:
        return None


def _gpu_name() -> str:
    import subprocess

    return subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def _wait_ready(process) -> bool:
    """Poll /health every 0.1 s until vLLM answers or exits."""
    while process.poll() is None and _local_vllm("/health") is None:
        time.sleep(0.1)
    return process.poll() is None


def _watch_startup(
    process,
    emit,
    baseline: dict | None = None,
    ready_event="vllm_ready",
    headers: dict | None = None,
) -> None:
    """Record GPU, readiness, and the first request's engine timings from vLLM endpoints.

    `baseline` holds engine counters from earlier requests (snapshot warmup) so only the first
    real request is reported; `ready_event=None` skips the event when vLLM was already ready.
    `headers` go with the /metrics requests (the snapshot server's internal token).
    """
    from cold_start_timing import container_clock, engine_metrics

    emit("gpu", gpu=_gpu_name())

    if not _wait_ready(process):
        emit("vllm_exited", returncode=process.returncode)
        return
    if ready_event:
        emit(ready_event, **container_clock())

    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline and process.poll() is None:
        engine = engine_metrics(_local_vllm("/metrics", headers=headers) or "", baseline)
        if engine["requests"] >= 1:
            emit(
                "first_token_observed",
                poll_interval=0.1,
                **engine,
                cache_state=_cache_state(),
                **container_clock(),
            )
            return
        time.sleep(0.1)


def sync_tokenizer_config() -> str:
    """Return the pinned tokenizer/config directory on the Volume, fetching only missing files."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=TOKENIZER_REPO,
        revision=TOKENIZER_REVISION,
        allow_patterns=["*.json", "*.jinja", "*.txt"],
        local_dir=CONFIG_DIR,
    )


def vllm_command(config_dir: str) -> list[str]:
    """Return the vLLM server command shared by every deployment variant."""
    return [
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
        "--download-dir", HF_HUB_DIR,
        "--max-model-len", "4096",
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.85",
        "--reasoning-parser", "qwen3",
        # The tutor binds a load_skill tool on every request.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--override-generation-config", '{"max_new_tokens": 512}',
        "--enforce-eager",
    ]


@app.function(
    image=image.env({"HF_HOME": "/cache/huggingface"}).add_local_python_source(
        "cold_start_timing"
    ),
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
    import threading

    from cold_start_timing import container_clock, timing_line

    def emit(event: str, **fields) -> None:
        print(timing_line(event, **fields), flush=True)

    emit("serve_entered", **container_clock())
    emit("cache_state", **_cache_state())

    config_dir = sync_tokenizer_config()
    emit("tokenizer_config_ready")

    process = subprocess.Popen(
        vllm_command(config_dir),
        env={**os.environ, "TRITON_CACHE_DIR": TRITON_CACHE_DIR},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    spawned = time.monotonic()
    emit("vllm_spawned", spawned_mono=round(spawned, 4))
    threading.Thread(target=_forward_vllm_output, args=(process, spawned), daemon=True).start()
    threading.Thread(target=_watch_startup, args=(process, emit), daemon=True).start()