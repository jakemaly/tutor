"""Run the student and judge as local vLLM OpenAI-compatible servers on the training GPU.

vLLM lives in its own virtual environment so its torch/transformers pins never fight the
training stack's. Each server gets a fixed slice of GPU memory (`gpu_memory_utilization`), so
start the servers before loading the tutor.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

VLLM_VERSION = "0.32.0"  # same as deployment/modal_qwen.py


def ensure_vllm_env(env_dir: Path, version: str = VLLM_VERSION) -> Path:
    """Create (once) a venv with vLLM and return the path to its `vllm` executable."""
    env_dir = Path(env_dir)
    vllm_bin = env_dir / "bin" / "vllm"
    if vllm_bin.exists():
        out = subprocess.run(
            [str(env_dir / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
        )
        if out.stdout.strip() == version:
            return vllm_bin
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", "--python", "3.12", "--seed", str(env_dir)], check=True)
        cmd = [uv, "pip", "install", "--python", str(env_dir / "bin" / "python")]
        subprocess.run(cmd + [f"vllm=={version}"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
        pip = [str(env_dir / "bin" / "python"), "-m", "pip", "install"]
        subprocess.run(pip + [f"vllm=={version}"], check=True)
    return vllm_bin


@dataclass
class VLLMServer:
    name: str
    process: subprocess.Popen | None  # None: a server we found already running on the port
    base_url: str
    served_model_name: str
    log_path: Path

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()

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
        *(extra_args or []),
    ]  # fmt: skip
    env = dict(os.environ)
    if hf_home:
        env["HF_HOME"] = hf_home
    log = open(log_path, "w")  # noqa: SIM115 - the server process owns it
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    server = VLLMServer(name, process, base_url, name, log_path)
    wait_until_ready(server, timeout)
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
    server.stop()
    raise TimeoutError(f"vLLM {server.name} not ready after {timeout}s:\n{server.log_tail()}")


def export_env(server: VLLMServer, prefix: str) -> None:
    """Point <PREFIX>_BASE_URL / _MODEL at a local server so `model_from_section` finds it."""
    os.environ[f"{prefix}_BASE_URL"] = server.base_url
    os.environ[f"{prefix}_MODEL"] = server.served_model_name
    os.environ[f"{prefix}_API_KEY"] = "EMPTY"


def start_student_and_judge(cfg, vllm_bin: Path, log_dir: Path) -> list[VLLMServer]:
    """Serve the configured student and judge locally and export their env variables."""
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
