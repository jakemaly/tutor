"""Serve the tutor's model on Modal from a GPU memory snapshot.

The model is unsloth/Qwen3.5-9B: higher-precision Safetensors weights that vLLM quantizes to
4-bit with BitsAndBytes while loading (in-flight quantization, through vllm-bnb-plugin).
Only Safetensors files are downloaded, and model_files.check_safetensors_only() refuses to
start if weights in any other format are present.

A restored cold start skips vLLM's imports, model load and quantization, and CUDA/kernel
setup; see docs/cold-start-benchmark.md for the measurements.

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
import model_files
import vllm_route_guard

app = modal.App("socratic-tutor-qwen")

# Unquantized (BF16) Safetensors, quantized to 4-bit by BitsAndBytes while loading. Weights,
# config, and image processor come from here.
MODEL_REPO = "unsloth/Qwen3.5-9B"
MODEL_REVISION = "005429cee5cb648998cf2b70eebdd83175989c9a"

# The tokenizer and chat template stay the official Qwen ones the tutor was validated with
# (vLLM resolves the chat template from the tokenizer path).
TOKENIZER_REPO = "Qwen/Qwen3.5-9B"
TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SERVED_MODEL_NAME = "socratic-tutor"
BASE_MODEL_NAME = "socratic-tutor-base"
ADAPTER_REVISION = "83e64b4eeb0ec550a0dd839c290a59fcfb802c0f"
ADAPTER_PATH = f"/cache/runs/jaylen-{ADAPTER_REVISION[:7]}"
ADAPTER_SHA256 = {
    "adapter_config.json": "e53585c3c01db38e7ae978be695d8d793167d2dbbe3285c8bcd55c16b186ed2d",
    "adapter_model.safetensors": "d55a7bc8b180e89704a9dd1721aebba34080ac3b417e3cbebd01f4572e4661c5",
}

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
    # vLLM 0.32.0 provides BitsAndBytes through this out-of-tree plugin; vLLM's own CI tests
    # it with `vllm-bnb-plugin >= 0.0.1`. The plugin requires bitsandbytes>=0.48.1.
    .pip_install("vllm-bnb-plugin==0.0.3", "bitsandbytes==0.50.2")
    .run_commands("python -c 'import bitsandbytes, vllm_bnb_plugin'")
)

model_cache = modal.Volume.from_name(
    "socratic-tutor-model-cache",
    create_if_missing=True,
)


HF_HUB_DIR = "/cache/huggingface/hub"
MODEL_SNAPSHOT = f"{HF_HUB_DIR}/models--unsloth--Qwen3.5-9B/snapshots/{MODEL_REVISION}"
TOKENIZER_SNAPSHOT = (
    f"{HF_HUB_DIR}/models--Qwen--Qwen3.5-9B/snapshots/{TOKENIZER_REVISION}"
)
# vLLM compiles about 1,100 Triton kernel files while profiling Qwen3.5; keeping them on the
# Volume lets later cold containers load them instead of recompiling (about 60 s on L40S).
# Triton keys entries by kernel source, Triton version, and GPU target (L4 and L40S: sm_89).
TRITON_CACHE_DIR = "/cache/triton-cache"


def _cache_state() -> dict:
    """Report which model files and compiled-kernel caches exist before vLLM starts."""
    from pathlib import Path

    from cold_start_timing import directory_state

    return {
        "safetensors": model_files.safetensors_state(MODEL_SNAPSHOT),
        "tokenizer_snapshot": directory_state(TOKENIZER_SNAPSHOT),
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


def sync_model_files() -> str:
    """Return the pinned model snapshot on the Volume, downloading only missing files.

    Only Safetensors weights, their index, and config/tokenizer/processor files are fetched
    (about 19.3 GB of BF16 weights on the first run). Raises if the directory holds weights in
    any other format, so vLLM never sees them.
    """
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=list(model_files.ALLOW_PATTERNS),
        cache_dir=HF_HUB_DIR,
    )
    model_files.check_safetensors_only(model_dir)
    return model_dir


def validate_adapter(path: str = ADAPTER_PATH) -> None:
    """Fail closed if the pinned, converted adapter is missing or incomplete."""
    import hashlib
    from pathlib import Path

    for name, expected in ADAPTER_SHA256.items():
        digest = hashlib.sha256()
        with (Path(path) / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Adapter file failed integrity check: {name}")


def vllm_command(model_dir: str) -> list[str]:
    """Return the vLLM server command.

    There is no fixed output cap, and the tool-call flags let vLLM parse the tutor's
    load_skill tool calls.
    """
    return [
        "vllm",
        "serve",
        model_dir,
        # In-flight 4-bit quantization of the BF16 Safetensors (vllm-bnb-plugin). The plugin
        # sets the bitsandbytes load format, which reads `*.safetensors` when present.
        "--quantization", "bitsandbytes",
        "--tokenizer", TOKENIZER_REPO,
        "--tokenizer-revision", TOKENIZER_REVISION,
        "--served-model-name", BASE_MODEL_NAME,
        "--enable-lora",
        "--max-lora-rank", "32",
        "--max-loras", "1",
        "--lora-modules", f"{SERVED_MODEL_NAME}={ADAPTER_PATH}",
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


# Environment for the vLLM subprocess only, applied before it initializes torch.distributed
# (and so before warmup, sleep, and the snapshot). The Modal container's own environment is
# unchanged.
VLLM_ENV = {
    "TRITON_CACHE_DIR": TRITON_CACHE_DIR,
    "VLLM_SERVER_DEV_MODE": "1",
    # Modal's guidance for Torch compilation inside a snapshot.
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    # Workaround; root cause unconfirmed. One snapshot restore crashed when PyTorch's NCCL
    # heartbeat monitor polled vLLM's file-based store (FileStore under /tmp) and flock()
    # failed with "Bad file descriptor"; the throw from FileStore's lock destructor ended the
    # process. In torch 2.13.0 that 1-second poll runs only while TORCH_NCCL_DUMP_ON_TIMEOUT
    # is enabled, so this turns it off. Lost: flight-recorder dumps after a collective timeout
    # (diagnostics only on this single-GPU server). TORCH_NCCL_ENABLE_MONITORING stays on: it
    # does not stop the poll, only the kill on a hung watchdog, which we keep.
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "0",
}


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


serving_image = image.env({"HF_HOME": "/cache/huggingface"}).add_local_python_source(
    "cold_start_timing", "model_files", "vllm_route_guard"
)


@app.function(image=serving_image, volumes={"/cache": model_cache}, cpu=2, timeout=3600)
def download_model() -> None:
    """Fetch the pinned Safetensors snapshot onto the Volume without a GPU.

    Run `modal run deployment/modal_qwen.py::download_model` after changing MODEL_REVISION, so
    the first GPU cold start does not spend paid GPU time downloading about 19.3 GB.
    """
    model_dir = sync_model_files()
    model_cache.commit()
    print(f"Safetensors ready: {model_files.safetensors_state(model_dir)}", flush=True)


@app.cls(
    image=serving_image,
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
        validate_adapter()
        model_dir = sync_model_files()
        _emit("model_files_ready", **model_files.safetensors_state(model_dir))

        # Only this process and vLLM know the token; it is kept in memory and in the snapshot.
        token = secrets.token_urlsafe(32)
        self.internal_headers = {vllm_route_guard.INTERNAL_HEADER: token}
        # The `vllm` console script does not search /root, where Modal adds local modules.
        guard_dir = os.path.dirname(vllm_route_guard.__file__)
        python_path = os.pathsep.join(filter(None, [guard_dir, os.environ.get("PYTHONPATH")]))

        self.process = subprocess.Popen(
            vllm_command(model_dir),
            env={
                **os.environ,
                **VLLM_ENV,
                vllm_route_guard.TOKEN_ENV: token,
                "PYTHONPATH": python_path,
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