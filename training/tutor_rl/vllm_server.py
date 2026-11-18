"""Run the student and judge as local vLLM OpenAI-compatible servers on the training GPU.

vLLM lives in its own virtual environment so its torch/transformers pins never fight the
training stack's. Each server gets a fixed slice of GPU memory (`gpu_memory_utilization`), so
start the servers before loading the tutor.

GPU memory hygiene: `vllm serve` runs the model in a child EngineCore process, and that child
holds the GPU memory. Each server therefore runs in its own process group and is stopped as a
group, and `kill_stale_vllm` removes vLLM processes left over from a crashed or interrupted
start (otherwise each retry strands ~20 GB and the next start runs out of memory).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

VLLM_VERSION = "0.32.0"  # same as deployment/modal_qwen.py
GIB = 1024**3


def vllm_env(vllm_bin: Path | None = None) -> dict:
    """Environment for vLLM processes: only their own venv, never the kernel's.

    - PYTHONPATH / VIRTUAL_ENV point at the notebook kernel's site-packages; inherited, they make
      the venv's `import vllm` load the kernel's torch and fail.
    - PYTORCH_CUDA_ALLOC_CONF is set for the training process; vLLM manages its own memory.
    - FlashInfer's sampler is JIT-compiled at startup and fails on molab (CUDA header mismatch).
    """
    env = dict(os.environ)
    for key in ("PYTHONPATH", "VIRTUAL_ENV", "PYTORCH_CUDA_ALLOC_CONF"):
        env.pop(key, None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    if vllm_bin is not None:  # the venv's tools (ninja, etc.) first on PATH
        env["PATH"] = str(Path(vllm_bin).parent) + os.pathsep + env.get("PATH", "")
    return env


def ensure_vllm_env(env_dir: Path, version: str = VLLM_VERSION) -> Path:
    """Create (once) a venv with vLLM and return the path to its `vllm` executable."""
    env_dir = Path(env_dir)
    vllm_bin = env_dir / "bin" / "vllm"
    if vllm_bin.exists():
        out = subprocess.run(
            [str(env_dir / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            env=vllm_env(vllm_bin),
        )
        if out.stdout.strip() == version:
            return vllm_bin
    shutil.rmtree(env_dir, ignore_errors=True)  # missing or broken: rebuild from scratch
    uv = shutil.which("uv")
    if uv:
        subprocess.run(
            [uv, "venv", "--python", "3.12", "--seed", str(env_dir)], check=True, env=vllm_env()
        )
        cmd = [uv, "pip", "install", "--python", str(env_dir / "bin" / "python")]
        subprocess.run(cmd + [f"vllm=={version}"], check=True, env=vllm_env())
    else:
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
        pip = [str(env_dir / "bin" / "python"), "-m", "pip", "install"]
        subprocess.run(pip + [f"vllm=={version}"], check=True, env=vllm_env())
    return vllm_bin


# ----- GPU memory --------------------------------------------------------------------------


def gpu_memory() -> tuple[float, float] | None:
    """(used GiB, total GiB) of GPU 0 from nvidia-smi, or None if it is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.splitlines()[0]
        used, total = (float(x) / 1024 for x in out.split(","))
        return used, total
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def gpu_processes() -> list[dict]:
    """Processes holding GPU memory: pid, GiB, and command line (Linux)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    procs = []
    for line in out.splitlines():
        try:
            pid, mib = (x.strip() for x in line.split(","))
            procs.append({"pid": int(pid), "gib": float(mib) / 1024, "cmd": _cmdline(int(pid))})
        except ValueError:
            continue
    return procs


def gpu_report() -> str:
    """A Markdown table of GPU use per process, for the notebook and for error messages."""
    mem = gpu_memory()
    head = f"GPU memory: {mem[0]:.1f} / {mem[1]:.1f} GiB used\n\n" if mem else ""
    rows = [
        f"| {p['pid']} | {p['gib']:.1f} | `{(p['cmd'] or '?')[:90]}` |" for p in gpu_processes()
    ]
    if not rows:
        return head + "(no per-process data from nvidia-smi)"
    # In a container nvidia-smi often cannot see process ids; the total above is still right.
    return head + "| pid | GiB | command |\n| ---: | ---: | --- |\n" + "\n".join(rows)


def is_vllm_process(cmd: str) -> bool:
    low = cmd.lower()
    return "vllm" in low and "pytest" not in low


def kill_stale_vllm(keep_pids: set[int] | None = None, timeout: float = 30) -> list[int]:
    """Kill vLLM processes on the GPU (orphaned EngineCores, half-started servers).

    Only processes whose command line mentions vLLM are touched. Returns the killed pids.
    """
    keep = (keep_pids or set()) | {os.getpid()}
    victims = [
        p["pid"] for p in gpu_processes() if p["pid"] not in keep and is_vllm_process(p["cmd"])
    ]
    # Processes nvidia-smi cannot attribute (containers) are found through /proc instead.
    for proc in Path("/proc").glob("[0-9]*") if Path("/proc").exists() else []:
        pid = int(proc.name)
        if pid not in keep and pid not in victims and is_vllm_process(_cmdline(pid)):
            victims.append(pid)
    for pid in victims:
        _signal(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and any(_alive(pid) for pid in victims):
        time.sleep(1)
    for pid in victims:
        if _alive(pid):
            _signal(pid, signal.SIGKILL)
    return victims


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie (exited, not yet reaped) holds no GPU memory.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (OSError, IndexError):
        return True


def check_free_memory(name: str, fraction: float, margin_gib: float = 1.0) -> None:
    """Fail fast, with a per-process table, if the GPU cannot fit `fraction` of its memory."""
    mem = gpu_memory()
    if mem is None:
        return
    used, total = mem
    need = fraction * total + margin_gib
    if total - used < need:
        raise RuntimeError(
            f"Not enough free GPU memory to start vLLM {name}: needs {need:.1f} GiB "
            f"({fraction:.0%} of {total:.0f} GiB + {margin_gib:.0f} GiB margin), "
            f"{total - used:.1f} GiB free.\n\n{gpu_report()}\n\n"
            "If a stale vLLM process is listed, click 'Stop servers' and start again; "
            "if the tutor is loaded, start the servers before loading it (or restart the kernel)."
        )


# ----- servers -----------------------------------------------------------------------------


@dataclass
class VLLMServer:
    name: str
    process: subprocess.Popen | None  # None: a server we found already running on the port
    base_url: str
    served_model_name: str
    log_path: Path

    def stop(self, timeout: float = 60) -> None:
        """Stop the server and its EngineCore children (the whole process group)."""
        if self.process is None:
            return
        pgid = self.process.pid  # start_new_session=True: the group id is the server's pid
        for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 10)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            try:
                self.process.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
            # The parent is gone; give its children a moment, then make sure.
            time.sleep(2)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


def start_server(
    vllm_bin: Path,
    name: str,
    section: dict,
    log_dir: Path,
    *,
    hf_home: str | None = None,
    extra_args: list[str] | None = None,
    timeout: float = 1800,
) -> VLLMServer:
    """Start `vllm serve` for a config section (model, revision, port, memory) and wait for it."""
    port = int(section["port"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vllm-{name}.log"
    base_url = f"http://127.0.0.1:{port}/v1"
    if served_models(base_url) == [name]:  # e.g. a notebook cell that was re-run
        return VLLMServer(name, None, base_url, name, log_path)
    check_free_memory(name, float(section["gpu_memory_utilization"]))
    cmd = [
        str(vllm_bin), "serve", section["model"],
        "--revision", section["revision"],
        "--served-model-name", name,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--gpu-memory-utilization", str(section["gpu_memory_utilization"]),
        "--max-model-len", str(section.get("max_model_len", 8192)),
        "--max-num-seqs", str(section.get("max_num_seqs", 64)),
        "--generation-config", "vllm",  # use our sampling params, not the repo's defaults
        *section.get("vllm_args", []),  # e.g. ["--quantization", "fp8"] for the student
        *(extra_args or []),
    ]  # fmt: skip
    env = vllm_env(vllm_bin)
    if hf_home:
        env["HF_HOME"] = hf_home
    log = open(log_path, "w")  # noqa: SIM115 - the server process owns it
    process = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )
    server = VLLMServer(name, process, base_url, name, log_path)
    try:
        wait_until_ready(server, timeout)
    except BaseException:
        server.stop()  # never leave a half-started EngineCore holding GPU memory
        raise
    return server


def served_models(base_url: str) -> list[str] | None:
    """Model names a server on `base_url` reports, or None if nothing answers."""
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=5) as r:
            return [m["id"] for m in json.loads(r.read())["data"]]
    except (OSError, ValueError, KeyError):
        return None


def wait_until_ready(server: VLLMServer, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.process is not None and server.process.poll() is not None:
            raise RuntimeError(f"vLLM {server.name} exited:\n{server.log_tail()}")
        if served_models(server.base_url):
            return
        time.sleep(5)
    raise TimeoutError(f"vLLM {server.name} not ready after {timeout}s:\n{server.log_tail()}")


def export_env(server: VLLMServer, prefix: str) -> None:
    """Point <PREFIX>_BASE_URL / _MODEL at a local server so `model_from_section` finds it."""
    os.environ[f"{prefix}_BASE_URL"] = server.base_url
    os.environ[f"{prefix}_MODEL"] = server.served_model_name
    os.environ[f"{prefix}_API_KEY"] = "EMPTY"


def start_student_and_judge(cfg, vllm_bin: Path, log_dir: Path) -> list[VLLMServer]:
    """Serve the configured student and judge locally and export their env variables.

    If not every server is already up, leftover vLLM processes are killed first, so a fresh
    start always has the GPU memory its config asks for.
    """
    wanted = [("student", cfg.student)]
    if not cfg.judge.get("share_student", False):
        wanted.append(("judge", cfg.judge))
    up = all(served_models(f"http://127.0.0.1:{int(s['port'])}/v1") == [n] for n, s in wanted)
    if not up:
        kill_stale_vllm()
    servers = []
    try:
        student = start_server(vllm_bin, "student", cfg.student, log_dir)
        servers.append(student)
        export_env(student, cfg.student["env_prefix"])
        if cfg.judge.get("share_student", False):
            export_env(student, cfg.judge["env_prefix"])
        else:
            judge = start_server(vllm_bin, "judge", cfg.judge, log_dir)
            servers.append(judge)
            export_env(judge, cfg.judge["env_prefix"])
    except BaseException:
        for s in servers:
            s.stop()
        raise
    return servers
