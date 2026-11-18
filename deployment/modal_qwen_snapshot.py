"""Serve the tutor's model from a Modal GPU memory snapshot (proposed deployment).

A restored cold start skips vLLM's imports, GGUF parsing, model load, and CUDA/kernel setup;
see docs/cold-start-benchmark.md. This is a separate Modal app from modal_qwen.py, which
remains the fallback and the benchmark control. It uses the same image, Volume, GPU list,
CPU, memory, scaling policy, and vLLM command. The differences are only those that
snapshots require:

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

Modal creates a few snapshots per GPU worker type after each deploy; those cold starts take the
full startup path plus about 8 s for warmup, sleep, and capture.
"""

import os
import secrets
import time
import uuid

import modal
import vllm_route_guard
from modal_qwen import (
    TRITON_CACHE_DIR,
    _cache_state,
    _forward_vllm_output,
    _gpu_name,
    _local_vllm,
    _wait_ready,
    _watch_startup,
    image,
    model_cache,
    sync_tokenizer_config,
    vllm_command,
)

app = modal.App("socratic-tutor-qwen-snapshot")

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
        "cold_start_timing", "modal_qwen", "vllm_route_guard"
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
            [
                *vllm_command(config_dir),
                "--enable-sleep-mode",
                "--middleware=vllm_route_guard.RouteGuard",
            ],
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
            body = {"model": "socratic-tutor", **body}
            if _local_vllm("/v1/chat/completions", "POST", body, 300, internal) is None:
                raise RuntimeError("A warmup request failed.")
        self.baseline = engine_metrics(_local_vllm("/metrics", headers=internal) or "")
        _emit("warmup_done", requests=self.baseline["requests"])

        for path in ("/reset_prefix_cache", "/sleep?level=1"):
            if _local_vllm(path, "POST", timeout=300, headers=self.internal_headers) is None:
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
            args=(self.process, _emit, self.baseline, None, self.internal_headers),
            daemon=True,
        ).start()

    @modal.web_server(port=8000, startup_timeout=900, requires_proxy_auth=True)
    def serve(self) -> None:
        """vLLM is already listening on port 8000."""

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
