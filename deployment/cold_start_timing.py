"""Dependency-free helpers for measuring Modal cold starts.

The Modal container imports this module to emit startup events. The local benchmark imports
it to parse those events, classify streamed chunks, and summarize trials.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EVENT_PREFIX = "[cold-start]"
VLLM_LINE = re.compile(r"\[vllm \+(\d+\.\d+)s\] (.*)")
LOG_TASK_ID = re.compile(r"\bta-[0-9A-Za-z]+\b")

# Only these container variables are recorded; others (for example identity tokens) are secret.
MODAL_ID_VARIABLES = (
    "MODAL_TASK_ID",
    "MODAL_IMAGE_ID",
    "MODAL_REGION",
    "MODAL_CLOUD_PROVIDER",
)

# First occurrence of each vLLM 0.32.0 log message marks the end of a startup stage.
VLLM_MILESTONES = (
    ("cli_imported", re.compile(r"non-default args")),
    ("weights_loaded", re.compile(r"Loading weights took ([\d.]+) seconds")),
    (
        "model_loaded",
        re.compile(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds"),
    ),
    ("memory_profiled", re.compile(r"Memory profiling takes ([\d.]+) seconds")),
    (
        "engine_initialized",
        re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+)"),
    ),
    ("api_started", re.compile(r"Application startup complete")),
)


def timing_line(event: str, **fields: Any) -> str:
    """Format one container event with wall and monotonic clocks for log correlation."""
    record = {
        "event": event,
        "wall": round(time.time(), 4),
        "mono": round(time.monotonic(), 4),
        **{name.lower(): os.environ.get(name) for name in MODAL_ID_VARIABLES},
        **fields,
    }
    return f"{EVENT_PREFIX} {json.dumps(record, sort_keys=True)}"


def log_task_id(line: str) -> str | None:
    """Return the Modal container (task) ID that `modal app logs --show-container-id` adds."""
    match = LOG_TASK_ID.search(line)
    return match.group(0) if match else None


def parse_timing_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Read container events from log lines that may carry Modal prefixes.

    `log_task_id` comes from Modal's log prefix. It is the container that printed the line,
    even after a snapshot restore, when the process's own environment may be the snapshot's.
    """
    events = []
    for line in lines:
        prefix, marker, payload = line.partition(EVENT_PREFIX)
        if not marker:
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if task := log_task_id(prefix):
            event["log_task_id"] = task
        events.append(event)
    return events


def parse_vllm_milestones(lines: Iterable[str]) -> dict[str, dict[str, float | None]]:
    """Return the first offset (seconds after vLLM spawn) and reported duration per milestone."""
    milestones: dict[str, dict[str, float | None]] = {}
    for line in lines:
        match = VLLM_LINE.search(line)
        if not match:
            continue
        offset, message = float(match.group(1)), match.group(2)
        for name, pattern in VLLM_MILESTONES:
            found = pattern.search(message)
            if found and name not in milestones:
                reported = float(found.group(1)) if found.groups() else None
                milestones[name] = {"offset": offset, "reported_seconds": reported}
    return milestones


def received_network_bytes(net_dev: str) -> int:
    """Sum received bytes on non-loopback interfaces from /proc/net/dev text."""
    total = 0
    for line in net_dev.splitlines()[2:]:
        name, _, counters = line.partition(":")
        if name.strip() == "lo" or not counters.split():
            continue
        total += int(counters.split()[0])
    return total


def process_clock(self_stat: str, uptime: str, now_wall: float, ticks_per_second: int) -> dict:
    """Estimate kernel boot and process start wall times from /proc text.

    Under Modal's sandbox, "boot" is the sandbox kernel start; tick resolution is 10 ms.
    """
    boot_wall = now_wall - float(uptime.split()[0])
    # Fields after the parenthesized command start at field 3; starttime is field 22.
    start_ticks = int(self_stat.rsplit(")", 1)[1].split()[19])
    return {
        "boot_wall": round(boot_wall, 3),
        "process_start_wall": round(boot_wall + start_ticks / ticks_per_second, 3),
    }


def container_clock() -> dict[str, Any]:
    """Read sandbox boot, process start, and network counters inside the container."""
    try:
        clock = process_clock(
            Path("/proc/self/stat").read_text(),
            Path("/proc/uptime").read_text(),
            time.time(),
            os.sysconf("SC_CLK_TCK"),
        )
    except (OSError, ValueError, IndexError):
        clock = {}
    return {**clock, "rx_bytes": network_bytes()}


def network_bytes() -> int | None:
    """Return received non-loopback bytes, or None where /proc/net/dev is unavailable."""
    try:
        return received_network_bytes(Path("/proc/net/dev").read_text())
    except (OSError, ValueError, IndexError):
        return None


def directory_state(path: str | Path) -> dict[str, Any]:
    """Describe a cache directory by file count and total size without reading file contents."""
    root = Path(path)
    if not root.exists():
        return {"exists": False, "files": 0, "bytes": 0}
    files = [item for item in root.rglob("*") if item.is_file()]
    return {
        "exists": True,
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def file_state(path: str | Path) -> dict[str, Any]:
    """Describe one cached file, following Hugging Face snapshot symlinks."""
    target = Path(path)
    return {
        "exists": target.exists(),
        "bytes": target.stat().st_size if target.exists() else 0,
    }


def metric_total(metrics_text: str, name: str) -> float | None:
    """Sum one Prometheus sample across label sets, or return None when it is absent."""
    total = None
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        sample, _, value = line.rpartition(" ")
        if sample.split("{", 1)[0] == name:
            total = (total or 0.0) + float(value)
    return total


ENGINE_METRICS = {
    "requests": "vllm:time_to_first_token_seconds_count",
    "ttft_seconds": "vllm:time_to_first_token_seconds_sum",
    "queue_seconds": "vllm:request_queue_time_seconds_sum",
    "prefill_seconds": "vllm:request_prefill_time_seconds_sum",
    "prompt_tokens": "vllm:prompt_tokens_total",
}


def engine_metrics(metrics_text: str, baseline: dict[str, float] | None = None) -> dict:
    """Read vLLM's cumulative first-token counters, minus an earlier reading such as warmup."""
    baseline = baseline or {}
    values = {}
    for key, name in ENGINE_METRICS.items():
        total = metric_total(metrics_text, name)
        values[key] = None if total is None else round(total - baseline.get(key, 0.0), 6)
    values["requests"] = values["requests"] or 0
    return values


def classify_sse_data(data: str) -> str:
    """Label one OpenAI-compatible stream payload so role-only chunks are not mistaken for text."""
    if data.strip() == "[DONE]":
        return "done"
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return "unparsed"
    if chunk.get("error"):
        return "error"
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("tool_calls"):
            return "tool_call"
        if delta.get("content"):
            return "content"
        if delta.get("reasoning_content") or delta.get("reasoning"):
            return "reasoning"
        if choice.get("finish_reason"):
            return "finish"
        if delta.get("role"):
            return "role"
    return "usage" if chunk.get("usage") else "empty"


def classify_data_stream_line(line: str) -> tuple[str, Any]:
    """Label one assistant-ui data-stream line; answer text is kind "text" with nonempty text."""
    if not line.strip():
        return "keepalive", None
    code, separator, payload = line.partition(":")
    if not separator:
        return "unparsed", line
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return "unparsed", line
    if code == "0":
        return ("text" if isinstance(value, str) and value else "empty_text"), value
    if code == "aui-text-delta":
        text = value.get("textDelta") if isinstance(value, dict) else None
        return ("text" if text else "empty_text"), text
    kinds = {
        "b": "tool_call_begin",
        "c": "tool_call_delta",
        "a": "tool_result",
        "3": "error",
        "f": "step_start",
        "e": "step_finish",
        "d": "finish",
        "g": "reasoning",
    }
    return kinds.get(code, "other"), value


def summarize(values: Iterable[float]) -> dict[str, float] | None:
    """Return median and range for a set of trial durations."""
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }
