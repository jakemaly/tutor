"""The deployment serves Safetensors with in-flight 4-bit BitsAndBytes and keeps its settings.

modal_qwen is imported with a stand-in `modal` module, so the real vllm_command() and the
recorded image, class, and endpoint arguments are checked without Modal or a GPU.
"""

import importlib
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "modal_qwen.py"
MODEL_DIR = "/cache/huggingface/hub/models--unsloth--Qwen3.5-9B/snapshots/abc"


@pytest.fixture(scope="module")
def deployment():
    saved = sys.modules.get("modal")
    fake_modal = MagicMock()
    sys.modules["modal"] = fake_modal
    sys.modules.pop("modal_qwen", None)
    try:
        yield importlib.import_module("modal_qwen"), fake_modal
    finally:
        sys.modules.pop("modal_qwen", None)
        if saved is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = saved


def flags(command: list[str]) -> dict[str, str | None]:
    """Map each --flag to its value (None for switches)."""
    result, items = {}, command[3:]
    for i, item in enumerate(items):
        if item.startswith("--"):
            name, _, inline = item.partition("=")
            following = items[i + 1] if i + 1 < len(items) else None
            has_value = inline or (following is not None and not following.startswith("--"))
            result[name] = inline or (following if has_value else None)
    return result


def recorded(fake_modal, method: str) -> list:
    return [c for c in fake_modal.mock_calls if c[0].split(".")[-1] == method]


def test_serves_the_local_safetensors_snapshot_with_bitsandbytes(deployment):
    module, _ = deployment
    command = module.vllm_command(MODEL_DIR)
    options = flags(command)

    assert command[:3] == ["vllm", "serve", MODEL_DIR]
    assert options["--quantization"] == "bitsandbytes"
    # The plugin sets the bitsandbytes load format itself; no GGUF config or model tag.
    assert "--load-format" not in options
    assert "--hf-config-path" not in options
    assert "--revision" not in options
    assert not any("gguf" in part.lower() or ":Q4_K_M" in part for part in command)


def test_keeps_tutor_serving_settings(deployment):
    module, _ = deployment
    options = flags(module.vllm_command(MODEL_DIR))

    assert options["--served-model-name"] == "socratic-tutor"
    assert options["--tokenizer"] == "Qwen/Qwen3.5-9B"
    assert re.fullmatch(r"[0-9a-f]{40}", options["--tokenizer-revision"])
    assert options["--max-model-len"] == "auto"
    assert "--override-generation-config" not in options
    assert "--enable-auto-tool-choice" in options
    assert options["--tool-call-parser"] == "qwen3_xml"
    assert options["--reasoning-parser"] == "qwen3"
    assert options["--default-chat-template-kwargs"] == '{"enable_thinking": false}'
    assert "--enforce-eager" in options
    assert "--enable-sleep-mode" in options
    assert options["--middleware"] == "vllm_route_guard.RouteGuard"


def test_model_is_the_pinned_unsloth_safetensors_repo(deployment):
    module, _ = deployment
    assert module.MODEL_REPO == "unsloth/Qwen3.5-9B"
    assert re.fullmatch(r"[0-9a-f]{40}", module.MODEL_REVISION)
    assert module.MODEL_SNAPSHOT.endswith(
        f"models--unsloth--Qwen3.5-9B/snapshots/{module.MODEL_REVISION}"
    )


def test_image_installs_the_bnb_plugin_and_no_gguf_loader(deployment):
    _, fake_modal = deployment
    installed = [arg for call in recorded(fake_modal, "pip_install") for arg in call.args]

    assert "vllm==0.32.0" in installed
    assert "vllm-bnb-plugin==0.0.3" in installed
    assert "bitsandbytes==0.50.2" in installed
    assert not any("gguf" in arg.lower() for arg in installed)
    # No GGUF fallback anywhere in the deployment code.
    assert "gguf" not in SOURCE.read_text(encoding="utf-8").lower()


def test_vllm_subprocess_turns_off_only_the_nccl_store_poll(deployment):
    """Workaround for the restore crash in PyTorch's NCCL heartbeat-monitor store poll."""
    module, fake_modal = deployment
    env = module.VLLM_ENV

    assert env["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "0"
    # The hang-kill monitor stays on; in torch 2.13.0 it does not control the store poll.
    assert "TORCH_NCCL_ENABLE_MONITORING" not in env
    assert env["VLLM_SERVER_DEV_MODE"] == "1"
    assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert env["TRITON_CACHE_DIR"] == module.TRITON_CACHE_DIR

    # Applied to the vLLM subprocess after the container's environment, so it wins; never
    # set on the Modal image or container.
    source = SOURCE.read_text(encoding="utf-8")
    assert source.index("**os.environ,") < source.index("**VLLM_ENV,")
    image_env = [arg for call in recorded(fake_modal, "env") for arg in call.args]
    assert not any(key.startswith("TORCH_NCCL") for mapping in image_env for key in mapping)
    assert 'os.environ["TORCH_NCCL' not in source


def test_keeps_scaling_snapshot_memory_and_endpoint_settings(deployment):
    _, fake_modal = deployment
    (cls_call,) = recorded(fake_modal, "cls")
    (endpoint,) = recorded(fake_modal, "web_server")

    assert cls_call.kwargs["gpu"] == ["L4", "L40S"]
    assert cls_call.kwargs["memory"] == 16384
    assert cls_call.kwargs["min_containers"] == 0
    assert cls_call.kwargs["scaledown_window"] == 60
    assert cls_call.kwargs["enable_memory_snapshot"] is True
    assert cls_call.kwargs["experimental_options"] == {"enable_gpu_snapshot": True}
    assert endpoint.kwargs["label"] == "socratic-tutor-qwen-serve"
    assert endpoint.kwargs["requires_proxy_auth"] is True
    assert endpoint.kwargs["startup_timeout"] == 900
