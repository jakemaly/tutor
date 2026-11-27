"""Package a run for download: adapter, config, data version, metrics, rollouts, eval, provenance.

Optimizer states and merged weights are left out of the zip (large, and not needed to serve or
compare the checkpoint); push those to the Hugging Face Hub instead if you need them.
"""

import importlib.metadata as md
import json
import platform
import subprocess
import zipfile
from pathlib import Path

PACKAGES = (
    "torch", "transformers", "peft", "accelerate", "datasets", "safetensors",
    "flash-linear-attention", "marimo",
)  # fmt: skip


def provenance(cfg, repo_dir: Path | None = None, extra: dict | None = None) -> dict:
    def version(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "not installed"

    info = {
        "run_name": cfg.run_name,
        "models": {
            role: {"model": section["model"], "revision": section["revision"]}
            for role, section in (
                ("tutor", cfg.tutor),
                ("student", cfg.student),
                ("judge", cfg.judge),
            )
        },
        "dataset": {k: cfg.data[k] for k in ("dataset", "dataset_config", "revision")},
        "packages": {p: version(p) for p in PACKAGES},
        "python": platform.python_version(),
    }
    if repo_dir is not None:
        try:
            info["repo_commit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            info["repo_commit"] = "unavailable"
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = (
                f"{props.name}, {props.total_memory / 1024**3:.0f} GB, CUDA {torch.version.cuda}"
            )
    except ImportError:
        pass
    info.update(extra or {})
    return info


def write_bundle(run_dir: Path, info: dict, zip_path: Path | None = None) -> Path:
    """Zip everything a reviewer or the serving step needs from `run_dir`."""
    run_dir = Path(run_dir)
    (run_dir / "provenance.json").write_text(json.dumps(info, indent=2))
    zip_path = Path(zip_path or run_dir.with_suffix(".zip"))
    skip_suffixes = {".pt", ".log"}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.rglob("*")):
            rel = path.relative_to(run_dir)
            if path.is_dir() or path.suffix in skip_suffixes or "merged" in rel.parts:
                continue
            if rel.parts[0] == "checkpoints" and "adapter" in rel.parts:
                continue  # intermediate adapters stay on disk/Hub; `final/` is in the zip
            zf.write(path, Path(run_dir.name) / rel)
    return zip_path


def save_merged(adapter_dir: Path, tutor_cfg: dict, out_dir: Path, device: str = "cuda") -> Path:
    """Base + adapter merged into one bf16 text-only checkpoint (for serving without LoRA)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        tutor_cfg["model"], revision=tutor_cfg["revision"], dtype=torch.bfloat16,
        device_map={"": device},
    )  # fmt: skip
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    del merged, base
    torch.cuda.empty_cache()
    return Path(out_dir)


def adapter_for_vllm(adapter_dir: Path, out_dir: Path) -> Path:
    """Copy a saved adapter with its weights renamed to the layout vLLM expects.

    Training loads Qwen3.5 text-only (`model.layers.*`), but vLLM serves the full checkpoint,
    whose language model sits under `model.language_model.layers.*`. Without the rename vLLM
    would not match the adapter to any layer and would quietly serve the base model.
    (deployment/modal_qwen_trained.py has a copy of this for the Modal container.)
    """
    import shutil

    from safetensors.torch import load_file, save_file

    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    old, new = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."
    renamed = {(new + k[len(old):] if k.startswith(old) else k): v for k, v in weights.items()}
    assert any(k.startswith(new) for k in renamed), "no LoRA weights for the language model"
    save_file(renamed, str(out_dir / "adapter_model.safetensors"))
    shutil.copy(adapter_dir / "adapter_config.json", out_dir / "adapter_config.json")
    return out_dir


HUB_IGNORE = ["*.pt", "*.log", "merged/**", "logs/**"]


def push_to_hub(
    folder: Path,
    repo_id: str,
    token: str | None = None,
    private: bool = True,
    ignore_patterns: list[str] | None = None,
) -> str:
    """Upload a folder to a (private by default) model repo; returns the commit URL."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        ignore_patterns=HUB_IGNORE if ignore_patterns is None else ignore_patterns,
    )
    return commit.commit_url if hasattr(commit, "commit_url") else str(commit)
