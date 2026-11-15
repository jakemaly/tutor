"""Hub sync picks the right files and the right checkpoint (HfApi faked, no network)."""

from concurrent.futures import Future
from pathlib import Path

import pytest

hf = pytest.importorskip("huggingface_hub")

from tutor_rl import hub  # noqa: E402


class FakeApi:
    uploads: list[dict] = []
    files: list[str] = []

    def __init__(self, token=None):
        pass

    def create_repo(self, *args, **kwargs):
        pass

    def upload_folder(self, *, folder_path, run_as_future=False, **kwargs):
        staged = sorted(
            p.relative_to(folder_path).as_posix()
            for p in Path(folder_path).rglob("*")
            if p.is_file()
        )
        FakeApi.uploads.append({"files": staged, **kwargs})
        future = Future()
        future.set_result(None)
        return future

    def list_repo_files(self, repo_id):
        return FakeApi.files


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    monkeypatch.setattr(hf, "HfApi", FakeApi)
    FakeApi.uploads, FakeApi.files = [], []


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_checkpoint_upload_stages_checkpoint_and_run_files(tmp_path) -> None:
    run = tmp_path / "run"
    for rel in [
        "checkpoints/step-00010/adapter/adapter_model.safetensors",
        "checkpoints/step-00010/optimizer.pt",
        "checkpoints/step-00005/optimizer.pt",  # older: already uploaded
        "metrics.jsonl",
        "rollouts/step-00010.jsonl",
        "logs/vllm-student.log",
        "data/train.jsonl",  # goes to the dataset repo, not here
    ]:
        _touch(run / rel)
    sync = hub.HubSync("socratictutor/run", token="t")
    sync.upload_checkpoint(run, run / "checkpoints" / "step-00010").result()
    sync.wait()
    (upload,) = FakeApi.uploads
    assert upload["files"] == [
        "checkpoints/step-00010/adapter/adapter_model.safetensors",
        "checkpoints/step-00010/optimizer.pt",
        "logs/vllm-student.log",
        "metrics.jsonl",
        "rollouts/step-00010.jsonl",
    ]
    assert upload["commit_message"] == "Save checkpoints/step-00010"
    assert not list(tmp_path.glob(".hub-staging-run/*")), "staging copy is removed"


def test_latest_hub_checkpoint_needs_optimizer_state() -> None:
    FakeApi.files = [
        "checkpoints/step-00010/trainer_state.json",
        "checkpoints/step-00010/optimizer.pt",
        "checkpoints/step-00020/trainer_state.json",  # upload cut off before optimizer.pt
        "final/trainer_state.json",
    ]
    assert hub.latest_hub_checkpoint("socratictutor/run") == ("checkpoints/step-00010", 10)
    FakeApi.files = []
    assert hub.latest_hub_checkpoint("socratictutor/run") is None
