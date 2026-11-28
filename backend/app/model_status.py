"""Expose allowlisted Modal startup milestones and service-level statistics."""

import asyncio
import json
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

PROVIDERS = {
    "CLOUD_PROVIDER_OCI": "Oracle Cloud",
    "CLOUD_PROVIDER_AWS": "AWS",
    "CLOUD_PROVIDER_GCP": "Google Cloud",
}
STAGES = {
    "serve_entered": ("starting", "Modal is starting a container"),
    "model_files_ready": ("loading", "Model files are available on the Modal Volume"),
    "vllm_spawned": ("loading", "Starting the vLLM server"),
    "vllm_ready": ("warming", "vLLM is ready; running warmup requests"),
    "warmup_done": ("warming", "Warmup requests completed"),
    "sleeping": ("creating_snapshot", "Creating a GPU memory snapshot"),
    "awake": ("ready", "Model is ready to answer"),
    "snapshot_restoring": ("restoring", "Restoring GPU memory snapshot"),
    "vllm_exited": ("failed", "Model server exited during startup"),
}
SHARD_PROGRESS = re.compile(r"Loading safetensors checkpoint shards:.*?\|\s*(\d+)/(\d+)")
GPU_INFO = re.compile(r"^(NVIDIA [A-Za-z0-9 -]+)(?:,\s*(\d+) MiB)?$")
VLLM_MILESTONES = (
    ("Initializing a V1 LLM engine", "Initializing the vLLM engine"),
    ("Loading model from scratch", "Loading model weights"),
    ("Model loading took", "Model weights loaded onto the GPU"),
    (
        "init engine (profile, create kv cache, warmup model)",
        "GPU profiling and cache setup completed",
    ),
    ("Loaded new LoRA adapter", "Trained tutor adapter loaded"),
)


def status_from_logs(entries: list[Any]) -> dict[str, Any]:
    """Summarize one container without sending raw logs or attributing concurrent workers."""
    containers: dict[str, dict[str, Any]] = {}
    for entry in entries:
        task = next((i for i in entry.context_ids if i.startswith("ta-")), None)
        if not task:
            continue
        text = entry.message.strip()
        event: dict[str, Any] = {}
        label = None
        stage = None
        marker = "[cold-start] "
        if text.startswith(marker):
            try:
                event = json.loads(text[len(marker) :])
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
        elif "Restoring Function from memory snapshot" in text:
            event = {"event": "snapshot_restoring"}
        elif "Creating GPU memory snapshot for Function" in text:
            event = {"event": "serve_entered"}
        elif re.match(r"^\[vllm \+[\d.]+s\]", text):
            match = SHARD_PROGRESS.search(text)
            if match:
                completed, total = map(int, match.groups())
                if 0 <= completed <= total <= 100 and total:
                    stage, label = (
                        "loading",
                        f"Loading model weights: {completed} of {total} shards",
                    )
            elif "non-default args:" in text and "'quantization': 'bitsandbytes'" in text:
                stage, label = "loading", "vLLM configured for BitsAndBytes 4-bit"
                event = {"event": "quantization"}
            else:
                for phrase, description in VLLM_MILESTONES:
                    if phrase in text:
                        stage, label = "loading", description
                        break
            if label is None:
                continue
        else:
            continue
        name = event.get("event")
        if event and (
            not isinstance(name, str)
            or name not in {*STAGES, "gpu", "restored", "cache_state", "quantization"}
        ):
            continue
        status = containers.setdefault(task, {"stage": "waiting"})
        if name in STAGES:
            stage, label = STAGES[name]
        elif name == "restored":
            creator = event.get("snapshot_task_id")
            if isinstance(creator, str) and creator != task:
                stage, label = "restoring", "Snapshot restored; waking the model"
            elif status["stage"] == "creating_snapshot":
                label = "Snapshot created; waking the model"
            elif status["stage"] != "restoring":
                stage, label = "starting", "Waking the model"
        elif name == "cache_state":
            cached = event.get("safetensors")
            if isinstance(cached, dict) and cached.get("exists") is True:
                files = cached.get("files")
                if isinstance(files, int) and not isinstance(files, bool) and files > 0:
                    status["weights_source"] = "Cached on Modal Volume"
        elif name == "quantization":
            status["quantization"] = "BitsAndBytes 4-bit"
        gpu = event.get("gpu")
        match = GPU_INFO.fullmatch(gpu) if isinstance(gpu, str) else None
        if match:
            status["gpu"] = match.group(1)
            if match.group(2):
                status["gpu_memory_mib"] = int(match.group(2))
        raw_provider = event.get("modal_cloud_provider")
        provider = PROVIDERS.get(raw_provider) if isinstance(raw_provider, str) else None
        if provider:
            status["provider"] = provider
        region = event.get("modal_region")
        if isinstance(region, str) and region.replace("-", "").isalnum() and len(region) <= 32:
            status["region"] = region
        if stage:
            status["stage"] = stage
        if label:
            status["detail"] = label
            timestamp = getattr(entry, "timestamp", None)
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            else:
                timestamp = None
            events = status.setdefault("events", [])
            if not events or events[-1]["label"] != label:
                events.append({"label": label, "at": timestamp})
                del events[:-8]
    if len(containers) != 1:
        return {"stage": "waiting"}
    return next(iter(containers.values()))


async def get_model_status(request_started: float | None) -> dict[str, Any]:
    """Fetch logs and live service counts independently; never block inference on telemetry."""
    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
        return {"stage": "unavailable"}
    if request_started is None:
        return {"stage": "waiting"}
    try:
        import modal

        now = datetime.now(UTC)
        oldest = max(now - timedelta(minutes=5), datetime.fromtimestamp(request_started, UTC))
        app_name = os.environ.get("MODAL_STATUS_APP", "socratic-tutor-qwen")

        async def logs():
            app = await modal.App.lookup.aio(app_name)
            entries = [entry async for entry in app.logs.fetch.aio(since=oldest)]
            return status_from_logs(entries)

        async def stats():
            method = modal.Cls.from_name(app_name, "SnapshotServer")().serve
            current = await method.get_current_stats.aio()
            return {
                "queued": current.backlog,
                "containers": current.num_total_runners,
                "running": current.num_running_inputs,
            }

        logs_result, stats_result = await asyncio.gather(
            asyncio.wait_for(logs(), timeout=5),
            asyncio.wait_for(stats(), timeout=5),
            return_exceptions=True,
        )
        result = logs_result if isinstance(logs_result, dict) else {"stage": "unavailable"}
        if isinstance(stats_result, dict):
            result["service"] = stats_result
        return result
    except Exception:
        return {"stage": "unavailable"}
