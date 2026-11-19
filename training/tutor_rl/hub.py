"""Hugging Face Hub as the durable store for runs (the team's rule: GitHub holds code, configs,
and small manifests; the Hub holds run outputs, checkpoints, logs, datasets, and models).

molab sessions end after 12 hours and do not keep files over 1 GB, so every checkpoint is
uploaded as soon as it is saved, and a new session resumes from the newest one on the Hub.

Layout of a run's model repo (private, in the `socratictutor` org by default):
    checkpoints/step-NNNNN/   adapter/, optimizer.pt, trainer_state.json, config.toml, ...
    final/                    adapter/ and config of the finished run
    metrics.jsonl  rollouts/  logs/  eval/  provenance.json
Datasets go to a separate dataset repo under <data_version>/.
"""

import re
import shutil
import threading
from concurrent.futures import Future
from pathlib import Path

DEFAULT_ORG = "socratictutor"
_RUN_FILES = ["metrics.jsonl", "rollouts/**", "eval/**", "provenance.json", "config.toml"]


class HubSync:
    """Uploads run snapshots in the background so training never waits on the network."""

    def __init__(self, repo_id: str, token: str | None = None, private: bool = True) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        self.futures: list[Future] = []
        self._lock = threading.Lock()

    def upload_run(self, run_dir: Path, *, include: list[str], message: str) -> Future:
        """Copy the selected files to a staging folder, then upload that frozen copy.

        Staging matters: metrics, rollouts and vLLM logs keep changing while the upload runs.
        """
        self.raise_errors()
        run_dir = Path(run_dir)
        stage = run_dir.parent / f".hub-staging-{run_dir.name}" / re.sub(r"\W+", "-", message)
        shutil.rmtree(stage, ignore_errors=True)
        for pattern in include:
            for src in run_dir.glob(pattern):
                if src.is_file():
                    dst = stage / src.relative_to(run_dir)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        future = self.api.upload_folder(
            folder_path=str(stage), repo_id=self.repo_id, commit_message=message, run_as_future=True
        )
        future.add_done_callback(lambda _f: shutil.rmtree(stage, ignore_errors=True))
        with self._lock:
            self.futures.append(future)
        return future

    def upload_checkpoint(self, run_dir: Path, checkpoint: Path) -> Future:
        rel = Path(checkpoint).relative_to(run_dir).as_posix()
        return self.upload_run(
            run_dir, include=[f"{rel}/**", *_RUN_FILES, "logs/**"], message=f"Save {rel}"
        )

    def raise_errors(self) -> None:
        """Surface a failed background upload instead of losing checkpoints silently."""
        with self._lock:
            done = [f for f in self.futures if f.done()]
            self.futures = [f for f in self.futures if not f.done()]
        for f in done:
            f.result()

    def wait(self) -> None:
        with self._lock:
            pending = list(self.futures)
        for f in pending:
            f.result()
        self.raise_errors()

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"


def latest_hub_checkpoint(repo_id: str, token: str | None = None) -> tuple[str, int] | None:
    """Newest checkpoints/step-N folder that has a trainer_state.json and optimizer state."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        files = set(HfApi(token=token).list_repo_files(repo_id))
    except RepositoryNotFoundError:
        return None
    steps = []
    for f in files:
        m = re.fullmatch(r"(checkpoints/step-(\d+))/trainer_state\.json", f)
        if m and f"{m.group(1)}/optimizer.pt" in files:
            steps.append((m.group(1), int(m.group(2))))
    return max(steps, key=lambda s: s[1]) if steps else None


def fetch_for_resume(repo_id: str, run_dir: Path, token: str | None = None) -> int:
    """Download the newest Hub checkpoint plus metrics/rollouts unless the local copy is newer.

    Returns the step that is now available locally (0 if there is nothing to resume).
    """
    from huggingface_hub import snapshot_download

    from tutor_rl.grpo import latest_checkpoint

    local = latest_checkpoint(run_dir)
    local_step = int(local.name.split("-")[1]) if local else 0
    remote = latest_hub_checkpoint(repo_id, token)
    if remote is None or remote[1] <= local_step:
        return local_step
    snapshot_download(
        repo_id,
        local_dir=str(run_dir),
        token=token,
        allow_patterns=[f"{remote[0]}/**", "metrics.jsonl", "rollouts/**"],
    )
    return remote[1]


def push_dataset(data_dir: Path, repo_id: str, token: str | None = None) -> str:
    """Upload train/eval JSONL + manifest to a dataset repo under <data_version>/."""
    import json

    from huggingface_hub import HfApi

    version = json.loads((Path(data_dir) / "manifest.json").read_text())["data_version"]
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(data_dir),
        path_in_repo=version,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Data version {version}",
    )
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/{version}"
