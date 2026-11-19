"""Measure genuine Modal cold starts through the tutor backend's /api/chat path.

Usage (from the repository root; see docs/cold-start-benchmark.md):

    python deployment/cold_start_benchmark.py run --label baseline --trials 3
    python deployment/cold_start_benchmark.py report docs/cold-start-results/*.json

`run` places a local timing relay between the backend and Modal. The client and relay share
one monotonic clock. Container events come from the app logs and are aligned by wall clock.
Results omit credentials, headers, request bodies, and the endpoint host.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cold_start_timing import (
    EVENT_PREFIX,
    classify_data_stream_line,
    classify_sse_data,
    log_task_id,
    parse_timing_lines,
    parse_vllm_milestones,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "socratic-tutor-qwen"
DEFAULT_PROMPT = "What does the derivative of a function tell us? Answer in two sentences."
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "te",
    "trailer",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}

# Stage boundaries telescope from request start to first answer text, so stages sum to TTFT.
# Each entry: (key, label, boundary that ends the stage, clock basis).
_START = (
    (
        "local_backend",
        "Local backend until upstream request",
        "relay_received",
        "same-clock",
    ),
)
_FULL_STARTUP = (
    (
        "provisioning",
        "Modal routing, GPU scheduling, container creation",
        "process_start",
        "cross-host",
    ),
    (
        "runtime_init",
        "Container runtime and function import",
        "serve_entered",
        "container",
    ),
    ("cache_scan", "Cache-state check (instrumentation)", "cache_state", "container"),
    (
        "tokenizer_sync",
        "Tokenizer/config sync (Volume + HF metadata)",
        "tokenizer_ready",
        "container",
    ),
    ("vllm_import", "vLLM CLI imports (torch, vLLM)", "cli_imported", "container"),
    (
        "engine_spawn",
        "Config resolution, engine spawn, CUDA init",
        "load_start",
        "container",
    ),
    ("model_load", "Model load (GGUF from Volume to GPU)", "model_loaded", "container"),
    (
        "engine_init",
        "Profiling, KV cache, warmup, kernel JIT",
        "engine_initialized",
        "container",
    ),
    ("api_startup", "API server startup until /health", "vllm_ready", "container"),
)
_SNAPSHOT_CAPTURE = (
    ("warmup", "Warmup requests before the snapshot", "warmup_done", "container"),
    ("sleep", "Prefix-cache reset and vLLM sleep (level 1)", "sleeping", "container"),
    (
        "snapshot",
        "Modal snapshot capture, then restore in place",
        "restored",
        "container",
    ),
)
_RESTORE = (
    (
        "provisioning_restore",
        "Modal routing, GPU scheduling, snapshot restore",
        "restored",
        "cross-host",
    ),
)
_END = (
    (
        "first_token",
        "Held request until first token in engine",
        "first_token",
        "container",
    ),
    ("return_path", "First token back to relay", "relay_first_content", "cross-host"),
    (
        "backend_stream",
        "Relay through backend to client text",
        "first_text",
        "same-clock",
    ),
)
_WAKE = (("wake_up", "vLLM wake_up: weights to GPU, KV cache", "awake", "container"),)
STAGE_SETS = {
    "none": _START + _FULL_STARTUP + _END,
    "created": _START + _FULL_STARTUP + _SNAPSHOT_CAPTURE + _WAKE + _END,
    "restored": _START + _RESTORE + _WAKE + _END,
}
STAGES = STAGE_SETS["none"]
# Events printed after a restore; their monotonic clock is anchored to the `restored` event.
RESTORE_PHASE_EVENTS = ("restored", "awake", "first_token_observed")
SNAPSHOT_LOG = re.compile(r"snapshot", re.IGNORECASE)


def read_env(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE lines, dropping inline comments, without printing values."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.split(" #", 1)[0].strip().strip("'\"")
    return values


def ntp_offset(server: str = "time.google.com") -> dict[str, float] | None:
    """Estimate local clock minus NTP time with one SNTP exchange."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(3)
            sent = time.time()
            sock.sendto(b"\x1b" + 47 * b"\0", (server, 123))
            data, _ = sock.recvfrom(48)
            received = time.time()
    except OSError:
        return None

    def stamp(offset: int) -> float:
        seconds, fraction = struct.unpack("!II", data[offset : offset + 8])
        return seconds - 2_208_988_800 + fraction / 2**32

    server_received, server_sent = stamp(32), stamp(40)
    offset = ((server_received - sent) + (server_sent - received)) / 2
    delay = (received - sent) - (server_sent - server_received)
    return {"local_minus_ntp": round(-offset, 4), "round_trip": round(delay, 4)}


class TimingRelay(ThreadingHTTPServer):
    """Forward backend model requests to Modal, timestamping stream events on the local clock."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], upstream: str, public_base: str):
        super().__init__(address, RelayHandler)
        parts = urlsplit(upstream)
        self.upstream_origin = f"{parts.scheme}://{parts.netloc}"
        self.upstream_host = parts.netloc
        self.public_base = public_base
        self.origin = 0.0
        self.records: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def now(self) -> float:
        return round(time.monotonic() - self.origin, 4)


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: TimingRelay

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Keep request lines (which may include redirect tokens) out of the console."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        self.forward()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        self.forward()

    def forward(self) -> None:
        relay = self.server
        record: dict[str, Any] = {
            "method": self.command,
            "path": self.path.split("?", 1)[0],
            "received": relay.now(),
            "first": {},
        }
        with relay.lock:
            relay.records.append(record)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Accept-Encoding"] = "identity"
        upstream = http.client.HTTPSConnection(relay.upstream_host, timeout=1800)
        try:
            upstream.request(self.command, self.path, body=body or None, headers=headers)
            record["forwarded"] = relay.now()
            response = upstream.getresponse()
            record["headers"] = relay.now()
            record["status"] = response.status
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP:
                    continue
                if key.lower() == "location":
                    value = value.replace(relay.upstream_origin, relay.public_base)
                self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            pending = ""
            while chunk := response.read1(65536):
                now = relay.now()
                record.setdefault("first_byte", now)
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                pending += chunk.decode("utf-8", errors="replace")
                *lines, pending = pending.split("\n")
                for line in lines:
                    if line.startswith("data:"):
                        kind = classify_sse_data(line[5:].strip())
                        record["first"].setdefault(kind, now)
            self.wfile.write(b"0\r\n\r\n")
            record["end"] = relay.now()
        except OSError as error:
            record["error"] = type(error).__name__
        finally:
            upstream.close()


def modal_cli(*args: str, timeout: int = 120) -> str:
    """Run the Modal CLI from this interpreter's environment."""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "modal", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"modal {' '.join(args[:2])} failed: {result.stderr[-500:]}")
    return result.stdout


def deployed_app_id(name: str) -> str:
    apps = json.loads(modal_cli("app", "list", "--json"))
    for app in apps:
        if app.get("description") == name and app.get("state") == "deployed":
            return app["app_id"]
    raise RuntimeError(f"No deployed Modal app named {name}.")


def running_containers(app_id: str) -> list[dict[str, Any]]:
    containers = json.loads(modal_cli("container", "list", "--app-id", app_id, "--json"))
    return [{key.lower().replace(" ", "_"): value for key, value in c.items()} for c in containers]


def wait_until_cold(app_id: str, timeout: float = 1200) -> float:
    """Block until the app has no running containers; return seconds waited."""
    started = time.monotonic()
    while running_containers(app_id):
        if time.monotonic() - started > timeout:
            raise RuntimeError("Containers did not scale down in time.")
        time.sleep(15)
    return round(time.monotonic() - started, 1)


def watch_containers(app_id: str, origin: float, stop: threading.Event, seen: dict) -> None:
    """Record when each container first appears in Modal's container list."""
    while not stop.is_set():
        try:
            for container in running_containers(app_id):
                container_id = container.get("container_id") or json.dumps(container)
                seen.setdefault(
                    container_id,
                    {**container, "seen": round(time.monotonic() - origin, 2)},
                )
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        stop.wait(3)


def stream_chat(backend: str, prompt: str, origin: float) -> dict[str, Any]:
    """POST one user message to /api/chat and timestamp each kind of stream event."""
    parts = urlsplit(backend)
    connection = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=1800)
    payload = json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]}
    )

    def now() -> float:
        return round(time.monotonic() - origin, 4)

    result: dict[str, Any] = {"first": {}, "answer_chars": 0, "errors": []}
    connection.request(
        "POST", "/api/chat", body=payload, headers={"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    result["headers"] = now()
    result["status"] = response.status
    pending = ""
    while chunk := response.read1(65536):
        pending += chunk.decode("utf-8", errors="replace")
        *lines, pending = pending.split("\n")
        for line in lines:
            kind, value = classify_data_stream_line(line)
            stamp = now()
            result["first"].setdefault(kind, stamp)
            if kind == "text":
                result["answer_chars"] += len(value)
            elif kind == "error":
                result["errors"].append(str(value)[:300])
    result["end"] = now()
    connection.close()
    return result


def fetch_trial_logs(app_id: str, since: datetime, t0_wall: float) -> tuple[list[str], set[str]]:
    """Return log lines from containers whose serve() started after the request was sent."""
    # A relative window avoids local-versus-UTC ambiguity in the CLI's datetime parsing.
    minutes = int((datetime.now(UTC) - since).total_seconds() // 60) + 3
    output = modal_cli(
        "app",
        "logs",
        app_id,
        "--since",
        f"{minutes}m",
        "--tail",
        "20000",
        "--show-container-id",
        timeout=300,
    )
    lines = output.splitlines()
    tasks = {
        event.get("log_task_id") or event["modal_task_id"]
        for event in parse_timing_lines(lines)
        if event.get("event") in ("serve_entered", "restored")
        and event.get("wall", 0) > t0_wall - 5
    }
    return [line for line in lines if log_task_id(line) in tasks], tasks


def snapshot_mode(trial: dict[str, Any]) -> str:
    """ "none" (no snapshot), "created" (this container made it), or "restored"."""
    names = {event["event"] for event in trial.get("container_events", [])}
    if "restored" not in names:
        return "none"
    return "created" if "serve_entered" in names else "restored"


def build_timeline(trial: dict[str, Any]) -> dict[str, float]:
    """Place every boundary on one axis: seconds after the client started the request."""
    clock = trial.get("clock") or {}
    t0_true = trial["t0_wall"] - (clock.get("local_minus_ntp") or 0.0)
    events = {event["event"]: event for event in trial.get("container_events", [])}
    timeline: dict[str, float] = {}
    records = trial.get("relay", [])
    if records:
        timeline["relay_received"] = records[0]["received"]
    answer = next((r for r in records if "content" in r.get("first", {})), None)
    if answer:
        timeline["relay_first_content"] = answer["first"]["content"]
    if "text" in trial["client"].get("first", {}):
        timeline["first_text"] = trial["client"]["first"]["text"]

    start_anchor = events.get("serve_entered")
    # The monotonic clock is not assumed continuous across a snapshot restore, so events after
    # the restore are placed relative to the `restored` event's own wall time.
    restore_anchor = events.get("restored")
    anchor = start_anchor or restore_anchor
    if not anchor:
        return timeline

    def at(event_name: str) -> float:
        event = events[event_name]
        base = restore_anchor if restore_anchor and event_name in RESTORE_PHASE_EVENTS else anchor
        return round(base["wall"] - t0_true + (event["mono"] - base["mono"]), 4)

    def at_mono(mono: float) -> float:
        return round(anchor["wall"] - t0_true + (mono - anchor["mono"]), 4)

    if start_anchor and "process_start_wall" in start_anchor:
        timeline["process_start"] = round(start_anchor["process_start_wall"] - t0_true, 4)
    names = {
        "serve_entered": "serve_entered",
        "cache_state": "cache_state",
        "tokenizer_ready": "tokenizer_config_ready",
        "vllm_ready": "vllm_ready",
        "warmup_done": "warmup_done",
        "sleeping": "sleeping",
        "restored": "restored",
        "awake": "awake",
        "first_token": "first_token_observed",
    }
    for name, key in names.items():
        if key in events:
            timeline[name] = at(key)
    spawned = events.get("vllm_spawned") if start_anchor else None
    milestones = trial.get("vllm_milestones", {})
    if spawned:
        for name, milestone in milestones.items():
            timeline[name] = at_mono(spawned["spawned_mono"] + milestone["offset"])
        loaded = milestones.get("model_loaded")
        if loaded and loaded.get("reported_seconds") is not None:
            timeline["load_start"] = round(timeline["model_loaded"] - loaded["reported_seconds"], 4)
    return timeline


def compute_stages(trial: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert the timeline into consecutive stage durations with their clock basis."""
    timeline = build_timeline(trial)
    stages, previous = [], 0.0
    for key, label, boundary, basis in STAGE_SETS[snapshot_mode(trial)]:
        end = timeline.get(boundary)
        duration = None if end is None or previous is None else round(end - previous, 3)
        stages.append({"key": key, "label": label, "seconds": duration, "basis": basis})
        previous = end
    return stages


def describe_trial(trial: dict[str, Any]) -> dict[str, Any]:
    """Derive the comparison fields: GPU, cache state, cold-start checks, and TTFT."""
    events = {event["event"]: event for event in trial.get("container_events", [])}
    mode = snapshot_mode(trial)
    cache = events.get("cache_state", {})
    first = events.get("first_token_observed", {})
    serve = events.get("serve_entered", {})
    restored = events.get("restored", {})
    tool_first = any(
        "tool_call" in record.get("first", {}) for record in trial.get("relay", [])
    ) or "tool_call_begin" in trial["client"].get("first", {})
    timeline = build_timeline(trial)
    if mode == "restored":
        # Network use from restore until vLLM was awake and serving.
        rx_before, rx_after = (
            restored.get("rx_bytes"),
            events.get("awake", {}).get("rx_bytes"),
        )
    else:
        rx_before, rx_after = (
            serve.get("rx_bytes"),
            events.get("vllm_ready", {}).get("rx_bytes"),
        )
    entry = serve or restored
    task_id = entry.get("log_task_id") or entry.get("modal_task_id")
    cold = {
        "no_containers_before_request": trial.get("containers_before") == 0,
        "new_task_id": trial.get("new_task_id"),
    }
    if mode == "restored":
        cold["restore_began_after_request"] = (timeline.get("restored") or -1) > 0
    else:
        cold["process_started_after_request"] = (timeline.get("process_start") or -1) > 0
    snapshot = None
    if mode != "none":
        snapshot = {
            "mode": mode,
            "snapshot_id": restored.get("snapshot_id"),
            "snapshot_task_id": restored.get("snapshot_task_id"),
            "restored_task_id": restored.get("log_task_id"),
            # A restore is genuine when a different container made the snapshot.
            "restored_from_other_container": (
                restored.get("snapshot_task_id") is not None
                and restored.get("log_task_id") is not None
                and restored["snapshot_task_id"] != restored["log_task_id"]
            ),
            "modal_log": trial.get("snapshot_log", []),
        }
    return {
        "label": trial["label"],
        "trial": trial["trial"],
        "task_id": task_id,
        "gpu": events.get("gpu", {}).get("gpu"),
        "ttft_seconds": timeline.get("first_text"),
        "snapshot": snapshot,
        "cold": cold,
        "cache": {
            "gguf_cached": cache.get("gguf", {}).get("exists"),
            "mmproj_cached": cache.get("mmproj", {}).get("exists"),
            "tokenizer_cached": cache.get("tokenizer_snapshot", {}).get("exists"),
            "kernel_cache_files_before": sum(
                cache.get(name, {}).get("files", 0)
                for name in ("triton_cache", "flashinfer_cache", "vllm_cache")
            ),
            "network_mb_until_ready": (
                round((rx_after - rx_before) / 1e6, 1)
                if rx_before is not None and rx_after is not None
                else None
            ),
        },
        "engine": {
            key: first.get(key)
            for key in (
                "ttft_seconds",
                "queue_seconds",
                "prefill_seconds",
                "prompt_tokens",
            )
        },
        "tool_call_before_answer": tool_first,
        "errors": trial["client"].get("errors", []),
        "stages": compute_stages(trial),
    }


def run(args: argparse.Namespace) -> None:
    env = read_env(ROOT / ".env")
    upstream = args.upstream or env.get("LLM_BASE_URL", "")
    if not upstream.startswith("https://"):
        sys.exit("Set LLM_BASE_URL in .env (or pass --upstream) to the Modal endpoint.")
    public_base = f"http://{args.relay_public_host}:{args.relay_port}"
    relay = TimingRelay((args.relay_bind, args.relay_port), upstream, public_base)
    threading.Thread(target=relay.serve_forever, daemon=True).start()

    app_id = deployed_app_id(args.app)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Every container used by an earlier saved trial counts as seen, so reuse is detected.
    previous_tasks = set(args.exclude_task or [])
    for saved in out_dir.glob("*.json"):
        previous_tasks |= set(json.loads(saved.read_text(encoding="utf-8")).get("task_ids", []))
    for number in range(args.first_trial, args.first_trial + args.trials):
        print(f"[{args.label} {number}] waiting for zero containers...", flush=True)
        waited = wait_until_cold(app_id)
        time.sleep(args.settle)
        containers_before = len(running_containers(app_id))
        clock = ntp_offset()
        relay.records.clear()
        seen: dict[str, Any] = {}
        stop = threading.Event()

        started_wall = datetime.now(UTC)
        t0_wall = time.time()
        origin = time.monotonic()
        relay.origin = origin
        watcher = threading.Thread(target=watch_containers, args=(app_id, origin, stop, seen))
        watcher.start()
        print(f"[{args.label} {number}] request sent", flush=True)
        try:
            client = stream_chat(args.backend, args.prompt, origin)
        finally:
            stop.set()
            watcher.join()
        if not relay.records:
            sys.exit(
                "The backend did not call the relay; start it with the benchmark compose file."
            )

        time.sleep(args.log_delay)
        logs, task_ids = fetch_trial_logs(app_id, started_wall, t0_wall)
        events = parse_timing_lines(logs)
        trial = {
            "label": args.label,
            "trial": number,
            "prompt": args.prompt,
            "started_utc": started_wall.isoformat(),
            "t0_wall": t0_wall,
            "clock": clock,
            "seconds_waited_for_scale_down": waited,
            "containers_before": containers_before,
            "containers_seen": list(seen.values()),
            "task_ids": sorted(task_ids),
            "new_task_id": len(task_ids) == 1 and not task_ids & previous_tasks,
            "client": client,
            "relay": relay.records[:],
            "container_events": events,
            "vllm_milestones": parse_vllm_milestones(logs),
            # Modal's own messages about snapshot creation or restore, for verification.
            "snapshot_log": [
                line[-300:]
                for line in logs
                if SNAPSHOT_LOG.search(line) and EVENT_PREFIX not in line
            ][:20],
        }
        previous_tasks |= task_ids
        trial["summary"] = describe_trial(trial)
        path = out_dir / f"{args.label}-{number}.json"
        path.write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
        print(f"[{args.label} {number}] TTFT {trial['summary']['ttft_seconds']} s -> {path}")
    relay.shutdown()


def report(args: argparse.Namespace) -> None:
    """Print Markdown tables for saved trials and per-label median/range."""
    trials = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.files]
    summaries = [describe_trial(trial) for trial in trials]
    modes = {mode: [] for mode in STAGE_SETS}
    for item in summaries:
        modes[(item["snapshot"] or {}).get("mode", "none")].append(item)
    for mode, items in modes.items():
        if not items:
            continue
        stage_keys = [key for key, *_ in STAGE_SETS[mode]]
        print(f"Snapshot mode: {mode}\n")
        print("| Trial | Container | GPU | Cache | " + " | ".join(stage_keys) + " | TTFT (s) |")
        print("|---" * (len(stage_keys) + 5) + "|")
        for item in items:
            cache = item["cache"]
            if mode == "restored":
                cache_text = "snapshot restore"
            else:
                cache_text = (
                    f"weights {'hit' if cache['gguf_cached'] else 'miss'}, "
                    f"kernels {cache['kernel_cache_files_before']} files"
                )
            seconds = {stage["key"]: stage["seconds"] for stage in item["stages"]}
            cells = [
                f"{seconds[key]:.1f}" if seconds[key] is not None else "n/a" for key in stage_keys
            ]
            gpu = (item["gpu"] or "?").split(",")[0]
            ttft = item["ttft_seconds"]
            ttft_text = f"{ttft:.1f}" if ttft is not None else "n/a"
            print(
                f"| {item['label']} {item['trial']} | `{(item['task_id'] or '?')[-8:]}` | {gpu} | "
                f"{cache_text} | " + " | ".join(cells) + f" | {ttft_text} |"
            )
        print()
    labels = sorted({item["label"] for item in summaries})
    for label in labels:
        group = [item for item in summaries if item["label"] == label]
        stage_keys = list(dict.fromkeys(s["key"] for item in group for s in item["stages"]))
        print(f"{label}: TTFT {summarize(item['ttft_seconds'] for item in group)}")
        for key in stage_keys:
            values = [
                stage["seconds"]
                for item in group
                for stage in item["stages"]
                if stage["key"] == key
            ]
            print(f"  {key}: {summarize(values)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Measure cold starts and save one JSON per trial")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--trials", type=int, default=3)
    run_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    run_parser.add_argument("--first-trial", type=int, default=1, help="Number of the first trial")
    run_parser.add_argument("--app", default=APP_NAME)
    run_parser.add_argument(
        "--upstream",
        help="Modal endpoint base URL (default: LLM_BASE_URL in .env); never saved",
    )
    run_parser.add_argument("--backend", default="http://127.0.0.1:8000")
    run_parser.add_argument("--relay-bind", default="0.0.0.0")
    run_parser.add_argument("--relay-port", type=int, default=8787)
    run_parser.add_argument("--relay-public-host", default="host.docker.internal")
    run_parser.add_argument("--settle", type=float, default=10, help="Seconds after scale-down")
    run_parser.add_argument("--log-delay", type=float, default=20, help="Seconds before log fetch")
    run_parser.add_argument("--exclude-task", action="append", help="Task IDs seen earlier")
    run_parser.add_argument("--out", default=str(ROOT / "docs" / "cold-start-results"))
    run_parser.set_defaults(handler=run)
    report_parser = commands.add_parser("report", help="Summarize saved trials")
    report_parser.add_argument("files", nargs="+")
    report_parser.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
