"""Which model files the deployment downloads and serves: Safetensors weights only.

The served model directory must contain Safetensors shards and no weights in any other
format. The BitsAndBytes loader (vllm-bnb-plugin) picks the first weight pattern that
matches, in the order `*.safetensors`, `*.bin`, `*.pt`. Downloading only Safetensors and
checking the directory before vLLM starts makes that choice explicit instead of implicit.
Standard library only, so the check runs in tests without Modal or vLLM.
"""

import fnmatch
import json
from pathlib import Path

# Safetensors shards and their index, plus config, tokenizer, processor, and chat-template files.
ALLOW_PATTERNS = ("*.safetensors", "*.json", "*.jinja", "*.txt")
# Weight formats that must not be present in the served directory.
OTHER_WEIGHT_PATTERNS = (
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.gguf",
    "*.h5",
    "*.msgpack",
    "*.npz",
    "*.onnx",
)
INDEX_FILE = "model.safetensors.index.json"


def _files(directory: str | Path) -> list[Path]:
    return sorted(path for path in Path(directory).rglob("*") if path.is_file())


def check_safetensors_only(directory: str | Path) -> list[str]:
    """Return the Safetensors shard names, or raise if the directory could load anything else.

    Fails when weights in another format are present, when there are no Safetensors files,
    or when the index lists a shard that is missing.
    """
    root = Path(directory)
    names = [str(path.relative_to(root)).replace("\\", "/") for path in _files(root)]
    others = [
        name
        for name in names
        if any(fnmatch.fnmatch(Path(name).name, p) for p in OTHER_WEIGHT_PATTERNS)
    ]
    if others:
        raise RuntimeError(f"Non-Safetensors weight files in {root}: {others}")
    shards = [name for name in names if name.endswith(".safetensors")]
    if not shards:
        raise RuntimeError(f"No Safetensors weights in {root}")
    index = root / INDEX_FILE
    if index.is_file():
        expected = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        missing = sorted(expected - set(shards))
        if missing:
            raise RuntimeError(f"Safetensors shards listed in {INDEX_FILE} are missing: {missing}")
    return shards


def safetensors_state(directory: str | Path) -> dict:
    """Describe the cached weights for timing logs; never raises."""
    try:
        shards = check_safetensors_only(directory)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        return {"exists": False, "reason": str(error)[:200]}
    root = Path(directory)
    return {
        "exists": True,
        "files": len(shards),
        # stat() follows the Hugging Face cache's symlinks to the stored blobs.
        "bytes": sum((root / name).stat().st_size for name in shards),
    }
