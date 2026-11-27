"""Stage the pinned private adapter for the snapshot server, without embedding its token."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from tutor_rl.export import adapter_for_vllm  # noqa: E402
from modal_qwen import ADAPTER_PATH, ADAPTER_REVISION, validate_adapter  # noqa: E402

ADAPTER_REPO = "socratictutor/tutor-rl-qwen35-9b-bigmath"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a token with read access to the private adapter.")
    with tempfile.TemporaryDirectory(prefix="tutor-adapter-") as directory:
        root = Path(directory)
        downloaded = snapshot_download(
            ADAPTER_REPO,
            revision=ADAPTER_REVISION,
            allow_patterns=["final/adapter/adapter_config.json", "final/adapter/adapter_model.safetensors"],
            local_dir=root / "download",
            token=token,
        )
        staged = adapter_for_vllm(Path(downloaded) / "final/adapter", root / "staged")
        validate_adapter(str(staged))
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "put", "--force",
             "socratic-tutor-model-cache", str(staged), ADAPTER_PATH.removeprefix("/cache")],
            check=True,
        )


if __name__ == "__main__":
    main()
