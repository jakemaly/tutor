"""The served model directory must contain Safetensors weights and nothing else."""

import json

import pytest
from model_files import ALLOW_PATTERNS, check_safetensors_only, safetensors_state

SHARDS = [f"model.safetensors-0000{i}-of-00002.safetensors" for i in (1, 2)]
INDEX = "model.safetensors.index.json"


def model_dir(tmp_path, *extra, shards=SHARDS, index=True):
    for name in (*shards, "config.json", "tokenizer.json", "preprocessor_config.json", *extra):
        (tmp_path / name).write_bytes(b"x" * 10)
    if index:
        weight_map = {f"layer.{i}.weight": name for i, name in enumerate(SHARDS)}
        (tmp_path / INDEX).write_text(json.dumps({"weight_map": weight_map}))
    return tmp_path


def test_safetensors_only_directory_passes_and_reports_its_shards(tmp_path):
    assert check_safetensors_only(model_dir(tmp_path)) == SHARDS
    assert safetensors_state(tmp_path) == {"exists": True, "files": 2, "bytes": 20}


@pytest.mark.parametrize(
    "extra",
    ["pytorch_model.bin", "model.pt", "Qwen3.5-9B-Q4_K_M.gguf", "mmproj.gguf", "sub/model.pth"],
)
def test_any_other_weight_format_is_refused(tmp_path, extra):
    if "/" in extra:
        (tmp_path / extra.split("/")[0]).mkdir()
    model_dir(tmp_path, extra)
    with pytest.raises(RuntimeError, match="Non-Safetensors weight files"):
        check_safetensors_only(tmp_path)
    assert safetensors_state(tmp_path)["exists"] is False


def test_missing_shard_listed_in_the_index_is_refused(tmp_path):
    model_dir(tmp_path, shards=SHARDS[:1])
    with pytest.raises(RuntimeError, match="missing"):
        check_safetensors_only(tmp_path)


def test_directory_without_safetensors_is_refused(tmp_path):
    model_dir(tmp_path, shards=[], index=False)
    with pytest.raises(RuntimeError, match="No Safetensors weights"):
        check_safetensors_only(tmp_path)


def test_download_patterns_exclude_every_other_weight_format():
    assert "*.safetensors" in ALLOW_PATTERNS
    assert not any(p.endswith((".bin", ".pt", ".pth", ".gguf")) for p in ALLOW_PATTERNS)
