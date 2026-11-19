"""Serve the tutor's model on Modal from a GPU memory snapshot.

A restored cold start skips vLLM's imports, GGUF parsing, model load, and CUDA/kernel setup;
see docs/cold-start-benchmark.md for the measurements.

- vLLM is started in a `snap=True` hook with `--enable-sleep-mode`, warmed up, and put to
  sleep (level 1: weights copied to host memory, KV cache freed) before Modal snapshots the
  container's CPU and GPU memory.
- After a restore, the `snap=False` hook wakes vLLM (weights back to the GPU, KV cache
  reallocated) before Modal routes requests to the web server.
- `VLLM_SERVER_DEV_MODE=1` is set for the vLLM process because vLLM exposes `/sleep` and
  `/wake_up` only in dev mode. Dev mode also adds administrative routes, so vLLM loads
  vllm_route_guard.RouteGuard: callers outside the container reach only the chat, model-list,
  and health routes. The rest, including sleep and wake, need a token that this container
  generates and keeps to itself.

A redeploy with new code or configuration needs new snapshots, and Modal may create several
per GPU type. Those cold starts take the full startup path plus about 8 s for warmup, sleep,
and capture.
"""

import os
import secrets
import time
import uuid

import modal
import vllm_route_guard

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


def _watch_startup(process, emit, baseline: dict, headers: dict) -> None:
    """Record the GPU and the first real request's engine timings from vLLM's metrics.

    `baseline` holds the engine counters after the warmup requests, so only the first real
    request is reported. `headers` carry the internal token that /metrics requires.
    """
    from cold_start_timing import container_clock, engine_metrics

    emit("gpu", gpu=_gpu_name())

    if not _wait_ready(process):
        emit("vllm_exited", returncode=process.returncode)
        return

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
    """Return the vLLM server command.

    There is no fixed output cap, and the tool-call flags let vLLM parse the tutor's
    load_skill tool calls.
    """
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
        "--download-dir", "/cache/huggingface/hub",
        # Use the longest context that fits this GPU.
        "--max-model-len", "auto",
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.85",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--enforce-eager",
        # Sleep and wake around the memory snapshot (routes enabled by VLLM_SERVER_DEV_MODE).
        "--enable-sleep-mode",
        # Hide dev-mode routes from callers outside the container.
        "--middleware=vllm_route_guard.RouteGuard",
    ]


# Short requests that run the text and tool-calling paths once before the snapshot, so any
# lazy initialization they trigger is captured. Generic prompts avoid any prefix-cache overlap
# with real requests, and the prefix cache is reset before sleeping as well.
WARMUP_REQUESTS = (
    {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16},
    {
        "messages": [{"role": "user", "content": "Which tool would help explain fractions?"}],
        "max_tokens": 16,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_topic",
                    "description": "Look up teaching notes for a topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"],
                    },
                },
            }
        ],
    },
)


def _emit(event: str, **fields) -> None:
    from cold_start_timing import timing_line

    print(timing_line(event, **fields), flush=True)


@app.cls(
    image=image.env({"HF_HOME": "/cache/huggingface"}).add_local_python_source(
        "cold_start_timing", "vllm_route_guard"
    ),
    gpu=["L4", "L40S"],
    cpu=2,
    # A reservation (guaranteed RAM), not a limit; all live tests passed with this value. One
    # awake restored L40S container used about 17.1 GiB, including an unknown amount of file
    # cache; peak usage was not measured. Memory above the reservation depends on the worker
    # having room and is not guaranteed. Modal bills the higher of reservation and usage.
    memory=16384,
    volumes={"/cache": model_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=600,
    startup_timeout=900,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class SnapshotServer:
    @modal.enter(snap=True)
    def start(self) -> None:
        """Start, warm up, and suspend vLLM; Modal snapshots the container after this returns."""
        import subprocess
        import threading

        from cold_start_timing import container_clock, engine_metrics

        _emit("serve_entered", phase="snapshot", **container_clock())
        _emit("cache_state", **_cache_state())
        config_dir = sync_tokenizer_config()
        _emit("tokenizer_config_ready")

        # Only this process and vLLM know the token; it is kept in memory and in the snapshot.
        token = secrets.token_urlsafe(32)
        self.internal_headers = {vllm_route_guard.INTERNAL_HEADER: token}
        # The `vllm` console script does not search /root, where Modal adds local modules.
        guard_dir = os.path.dirname(vllm_route_guard.__file__)
        python_path = os.pathsep.join(filter(None, [guard_dir, os.environ.get("PYTHONPATH")]))

        self.process = subprocess.Popen(
            vllm_command(config_dir),
            env={
                **os.environ,
                "TRITON_CACHE_DIR": TRITON_CACHE_DIR,
                "VLLM_SERVER_DEV_MODE": "1",
                vllm_route_guard.TOKEN_ENV: token,
                "PYTHONPATH": python_path,
                # Modal's guidance for Torch compilation inside a snapshot.
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        spawned = time.monotonic()
        _emit("vllm_spawned", spawned_mono=round(spawned, 4))
        threading.Thread(
            target=_forward_vllm_output, args=(self.process, spawned), daemon=True
        ).start()
        _emit("gpu", gpu=_gpu_name())
        if not _wait_ready(self.process):
            raise RuntimeError(f"vLLM exited during startup ({self.process.returncode}).")
        _emit("vllm_ready", **container_clock())

        internal = self.internal_headers
        for body in WARMUP_REQUESTS:
            body = {"model": SERVED_MODEL_NAME, **body}
            if _local_vllm("/v1/chat/completions", "POST", body, 300, internal) is None:
                raise RuntimeError("A warmup request failed.")
        self.baseline = engine_metrics(_local_vllm("/metrics", headers=internal) or "")
        _emit("warmup_done", requests=self.baseline["requests"])

        for path in ("/reset_prefix_cache", "/sleep?level=1"):
            if _local_vllm(path, "POST", timeout=300, headers=internal) is None:
                raise RuntimeError(f"vLLM did not accept {path}.")
        self.snapshot_id = uuid.uuid4().hex[:12]
        self.snapshot_task_id = os.environ.get("MODAL_TASK_ID")
        _emit("sleeping", snapshot_id=self.snapshot_id, **container_clock())

    @modal.enter(snap=False)
    def wake(self) -> None:
        """Runs after every restore (and after creation): put the weights back on the GPU."""
        import threading

        from cold_start_timing import container_clock

        _emit(
            "restored",
            snapshot_id=self.snapshot_id,
            snapshot_task_id=self.snapshot_task_id,
            **container_clock(),
        )
        if _local_vllm("/wake_up", "POST", timeout=300, headers=self.internal_headers) is None:
            raise RuntimeError("vLLM did not wake up after restore.")
        _emit("awake", **container_clock())
        threading.Thread(
            target=_watch_startup,
            args=(self.process, _emit, self.baseline, self.internal_headers),
            daemon=True,
        ).start()

    # The explicit label keeps the endpoint URL of the earlier `serve` function
    # (<workspace>--socratic-tutor-qwen-serve), so existing LLM_BASE_URL settings keep working.
    @modal.web_server(
        port=8000,
        startup_timeout=900,
        requires_proxy_auth=True,
        label="socratic-tutor-qwen-serve",
    )
    def serve(self) -> None:
        """vLLM is already listening on port 8000."""

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()