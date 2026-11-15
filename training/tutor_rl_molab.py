# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import importlib
    import importlib.metadata as md
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tarfile
    import time
    import urllib.request
    from dataclasses import replace
    from pathlib import Path

    REPO = "jakemaly/tutor"
    DEFAULT_REF = "46-ml-2-1-implement-the-pedagogicalrl-inspired-trainer-and-serve-its-checkpoint"

    # Training stack. torch comes from molab's GPU image; vLLM gets its own venv (see below).
    PINS = {
        "transformers": "5.23.0",
        "peft": "0.21.3",
        "accelerate": "1.15.0",
        "datasets": "5.2.0",
        "flash-linear-attention": "0.6.0",  # fast Gated DeltaNet kernels for Qwen3.5
    }


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutor RL: PedagogicalRL-style training for Qwen3.5-9B (issue #46)

    Trains the app's tutor model (**Qwen3.5-9B**, thinking off, bf16 LoRA) with multi-turn GRPO against a simulated student (**Llama-3.1-8B-Instruct**, as in the paper). A **Qwen2.5-14B-Instruct-AWQ** judge checks each dialogue for answer leaks. The reward is the student's solve rate after the dialogue, or −1 if the judge says the tutor gave the answer away. The code is in `training/tutor_rl/`; this notebook only wires it together.

    **Before you start**

    1. Turn on the GPU in molab's notebook specs. Expected: RTX PRO 6000, 96 GB.
    2. Accept the Llama 3.1 licence on Hugging Face, then paste a token with read access below. Add write access too if you want to push the results to the Hub.
    3. Run **smoke** first (a few minutes after downloads), then **molab**. molab sessions end after 12 hours: checkpoints are saved every `save_steps`, and *Resume* continues from the last one.

    **GPU layout:** student vLLM ~21 GB, judge vLLM ~15 GB, tutor training ~45 GB. The first run downloads about 45 GB of weights.
    """)
    return


@app.cell
def _(mo):
    config_choice = mo.ui.dropdown(["smoke", "molab"], value="smoke", label="Config")
    hf_token = mo.ui.text(
        value=os.environ.get("HF_TOKEN", ""), kind="password", label="Hugging Face token", full_width=True
    )
    repo_ref = mo.ui.text(value=DEFAULT_REF, label=f"{REPO} branch or commit", full_width=True)
    work_root = mo.ui.text(value=str(Path.home() / "tutor-rl"), label="Work directory", full_width=True)
    max_steps_override = mo.ui.number(start=0, stop=10_000, value=0, label="max_steps (0 = config)")
    share_judge = mo.ui.checkbox(False, label="Judge shares the student's server (saves ~15 GB)")
    mo.vstack([config_choice, hf_token, repo_ref, work_root, max_steps_override, share_judge])
    return (
        config_choice,
        hf_token,
        max_steps_override,
        repo_ref,
        share_judge,
        work_root,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Code and dependencies

    If this notebook runs from a checkout, the code next to it is used. Otherwise (for example when it was uploaded to molab on its own), the branch above is downloaded from GitHub and its exact commit is recorded.
    """)
    return


@app.cell
def _(mo, repo_ref, work_root):
    def _fetch_repo(ref: str, dest_root: Path) -> tuple[Path, str]:
        api = f"https://api.github.com/repos/{REPO}/commits/{ref}"
        with urllib.request.urlopen(api, timeout=30) as _r:
            _sha = json.loads(_r.read())["sha"]
        _dest = dest_root / f"tutor-{_sha[:12]}"
        if not (_dest / "training" / "tutor_rl").is_dir():
            _tmp = dest_root / f"tutor-{_sha[:12]}.tar.gz"
            urllib.request.urlretrieve(f"https://codeload.github.com/{REPO}/tar.gz/{_sha}", _tmp)
            with tarfile.open(_tmp) as _tf:
                _top = _tf.getnames()[0].split("/")[0]
                _tf.extractall(dest_root, filter="data")
            shutil.rmtree(_dest, ignore_errors=True)
            (dest_root / _top).rename(_dest)
            _tmp.unlink()
        return _dest, _sha

    _here = mo.notebook_dir()
    if _here is not None and (Path(_here) / "tutor_rl").is_dir():
        training_dir = Path(_here)
        repo_dir = training_dir.parent
        try:
            repo_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True
            ).stdout.strip() or "unknown (not a git checkout)"
        except OSError:
            repo_commit = "unknown (git not installed)"
    else:
        _root = Path(work_root.value)
        _root.mkdir(parents=True, exist_ok=True)
        repo_dir, repo_commit = _fetch_repo(repo_ref.value.strip(), _root)
        training_dir = repo_dir / "training"
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    mo.md(f"Code: `{training_dir}` at commit `{repo_commit}`")
    return repo_commit, repo_dir, training_dir


@app.cell
def _(mo):
    def _installed(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return ""

    if any(_installed(_n) != _v for _n, _v in PINS.items()):
        _reqs = [f"{_n}=={_v}" for _n, _v in PINS.items()]
        _uv = shutil.which("uv")
        _cmd = (
            [_uv, "pip", "install", "--python", sys.executable, *_reqs]
            if _uv
            else [sys.executable, "-m", "pip", "install", *_reqs]
        )
        subprocess.run(_cmd, check=True)
        importlib.invalidate_caches()
    _versions = {_n: _installed(_n) for _n in [*PINS, "torch"]}
    deps_ready = all(_installed(_n) == _v for _n, _v in PINS.items())
    assert deps_ready, f"Unexpected versions after install: {_versions}"

    import torch

    assert torch.cuda.is_available(), "No GPU: turn on the GPU in molab's notebook specs."
    _p = torch.cuda.get_device_properties(0)
    mo.md(
        f"GPU: **{_p.name}**, {_p.total_memory / 1024**3:.0f} GB, CUDA {torch.version.cuda}  \n"
        f"Packages: `{_versions}`"
    )
    return (deps_ready,)


@app.cell
def _(
    config_choice,
    deps_ready,
    hf_token,
    max_steps_override,
    mo,
    share_judge,
    training_dir,
    work_root,
):
    assert deps_ready
    from tutor_rl.config import load_config

    _base = load_config(training_dir / "configs" / f"{config_choice.value}.toml")
    _train = dict(_base.train)
    if max_steps_override.value:
        _train["max_steps"] = int(max_steps_override.value)
    cfg = replace(
        _base, train=_train, judge={**_base.judge, "share_student": bool(share_judge.value)}
    )
    run_dir = Path(work_root.value) / "outputs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if hf_token.value:
        os.environ["HF_TOKEN"] = hf_token.value

    _t = cfg.train
    mo.md(f"""
    **Run** `{cfg.run_name}` → `{run_dir}`

    | Role | Model | Revision |
    | --- | --- | --- |
    | Tutor (trained) | `{cfg.tutor['model']}` | `{cfg.tutor['revision'][:12]}` |
    | Student | `{cfg.student['model']}` | `{cfg.student['revision'][:12]}` |
    | Judge | `{cfg.judge['model'] if not cfg.judge['share_student'] else cfg.student['model'] + ' (shared)'}` | |

    {_t['max_steps']} steps × {_t['problems_per_step']} problems × {_t['episodes_per_problem']} episodes,
    up to {cfg.episode.max_turns} messages each, lr {_t['learning_rate']}, β {_t['beta']},
    LoRA r={cfg.tutor['lora_rank']}, checkpoint every {_t['save_steps']} steps.
    """)
    return cfg, run_dir


@app.cell
def _(cfg, mo, run_dir):
    from tutor_rl.data import prepare_data, read_problems

    _data_dir = run_dir / "data"
    if (_data_dir / "manifest.json").exists():
        data_manifest = json.loads((_data_dir / "manifest.json").read_text())
    else:
        data_manifest = prepare_data(cfg, _data_dir)
    train_problems = read_problems(_data_dir / "train.jsonl")
    eval_problems = read_problems(_data_dir / "eval.jsonl")
    mo.md(
        f"Data version `{data_manifest['data_version']}`: {len(train_problems)} train / "
        f"{len(eval_problems)} eval problems from `{data_manifest['dataset']}` "
        f"@ `{data_manifest['revision'][:12]}`. Example: *{train_problems[0].question}* → "
        f"`{train_problems[0].answer}`"
    )
    return data_manifest, eval_problems, train_problems


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Student and judge servers

    vLLM is installed once into its own venv (a few minutes), then the student and judge start as local OpenAI-compatible servers. They must start before the tutor loads, because each one reserves a fixed share of GPU memory. Re-running this cell reuses servers that are already up.
    """)
    return


@app.cell
def _(mo):
    start_servers = mo.ui.run_button(label="Start student and judge")
    start_servers
    return (start_servers,)


@app.cell
def _(cfg, mo, run_dir, start_servers, work_root):
    mo.stop(not start_servers.value, mo.md("*Click to start the servers.*"))
    from huggingface_hub import model_info

    from tutor_rl.vllm_server import ensure_vllm_env, start_student_and_judge

    try:  # fail fast on the gated Llama repo instead of deep inside vLLM's logs
        model_info(cfg.student["model"], revision=cfg.student["revision"], token=os.environ.get("HF_TOKEN"))
    except Exception as _e:
        raise RuntimeError(
            f"Cannot access {cfg.student['model']}: accept its licence on Hugging Face and "
            f"paste a token above. ({type(_e).__name__})"
        ) from None

    with mo.status.spinner(title="Installing vLLM (first run only)…"):
        _vllm_bin = ensure_vllm_env(Path(work_root.value) / "vllm-env")
    with mo.status.spinner(title="Starting student and judge (downloads weights on first run)…"):
        servers = start_student_and_judge(cfg, _vllm_bin, run_dir / "logs")
    mo.md("Running: " + ", ".join(f"`{_s.name}` at {_s.base_url}" for _s in servers))
    return (servers,)


@app.cell
def _(mo):
    stop_servers = mo.ui.run_button(label="Stop servers (frees ~36 GB)")
    stop_servers
    return (stop_servers,)


@app.cell
def _(mo, servers, stop_servers):
    mo.stop(not stop_servers.value)
    for _s in servers:
        _s.stop()
    mo.md("Servers stopped. Re-run the start cell to bring them back.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Tutor and a sanity-check dialogue

    One dialogue with the **base** tutor on the first eval problem, to check the prompts, the student, and the judge before spending hours on training.
    """)
    return


@app.cell
def _(cfg, deps_ready, mo, servers):
    assert deps_ready and servers  # GPU memory for vLLM is reserved first
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from tutor_rl.openai_chat import model_from_section
    from tutor_rl.policy import load_tutor

    with mo.status.spinner(title=f"Loading {cfg.tutor['model']} (bf16 + LoRA)…"):
        tutor = load_tutor(cfg.tutor)
    student = model_from_section(cfg.student)
    judge = model_from_section(cfg.judge)
    _trainable = sum(_p.numel() for _p in tutor.model.parameters() if _p.requires_grad)
    mo.md(f"Tutor loaded: {_trainable / 1e6:.1f}M trainable LoRA parameters.")
    return judge, student, tutor


@app.cell
def _(cfg, eval_problems, judge, mo, student, tutor):
    from tutor_rl.episode import run_episodes
    from tutor_rl.rewards import score_episodes

    tutor.use_adapter = False
    _ep = run_episodes(eval_problems[:1], tutor, student, cfg.episode)[0]
    _score = score_episodes([_ep], judge, cfg.reward)[0]
    tutor.use_adapter = True
    _rows = "\n\n".join(f"**{_t.speaker}:** {_t.content}" for _t in _ep.turns)
    mo.vstack([
        mo.md(f"**Problem** ({_ep.problem.answer}): {_ep.problem.question}"),
        mo.md(_rows),
        mo.md(
            f"**Student's final attempts:** {_ep.final_attempts}  \n"
            f"reward {_score.reward:.2f}, solve rate {_score.solve_rate:.2f}, "
            f"leaked {_score.leaked}, ended by tutor {_ep.ended_by_tutor}"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Train

    Each step logs to `metrics.jsonl` and `rollouts/step-*.jsonl` and updates the plot below. To stop early, interrupt the cell: the current step is saved as a checkpoint, and **Resume** picks it up later (also after a new molab session, if the work directory survived).
    """)
    return


@app.cell
def _(mo):
    resume = mo.ui.checkbox(True, label="Resume from the latest checkpoint if there is one")
    train_button = mo.ui.run_button(label="Train")
    mo.hstack([train_button, resume], justify="start")
    return resume, train_button


@app.function
def plot_metrics(rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    steps = [r["step"] for r in rows]
    for ax, key in zip(axes, ["reward", "solve_rate", "leak_rate", "kl"]):
        values = [r.get(key, float("nan")) for r in rows]
        ax.plot(steps, values, marker=".", alpha=0.5)
        if len(values) >= 5:
            smooth = [sum(values[max(0, i - 4) : i + 1]) / len(values[max(0, i - 4) : i + 1]) for i in range(len(values))]
            ax.plot(steps, smooth)
        ax.set_title(key)
        ax.set_xlabel("step")
    fig.tight_layout()
    plt.close(fig)
    return fig


@app.function
def read_metrics(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@app.cell
def _(
    cfg,
    judge,
    mo,
    resume,
    run_dir,
    student,
    train_button,
    train_problems,
    tutor,
):
    mo.stop(not train_button.value, mo.md("*Click **Train** to start.*"))
    from tutor_rl.grpo import GRPOTrainer

    if not resume.value:  # fresh run: set the previous attempt aside instead of mixing them
        _stamp = int(time.time())
        for _name in ("metrics.jsonl", "rollouts", "checkpoints", "final"):
            if (run_dir / _name).exists():
                (run_dir / _name).rename(run_dir / f"previous-{_stamp}-{_name}")
    trainer = GRPOTrainer(cfg, tutor, student, judge, train_problems, run_dir)
    _start = trainer.resume() if resume.value else 0
    _history = [_r for _r in read_metrics(run_dir) if _r["step"] <= _start]
    _total = cfg.train["max_steps"]
    _t0 = time.time()

    def _show(m):
        _history.append(m)
        _done = m["step"] - _start
        _eta = (time.time() - _t0) / _done * (_total - m["step"]) / 60
        mo.output.replace(mo.vstack([
            mo.md(
                f"Step **{m['step']}/{_total}**, reward {m['reward']:.3f}, solve {m['solve_rate']:.2f}, "
                f"leak {m['leak_rate']:.2f}, {m['step_seconds']:.0f} s/step, ~{_eta:.0f} min left, "
                f"peak GPU {m['max_gpu_gb']} GB"
            ),
            plot_metrics(_history),
        ]))

    mo.output.replace(mo.md(f"Starting at step {_start}…"))
    try:
        trained_adapter = trainer.train(on_step=_show) / "adapter"
    except BaseException:
        trainer.save(run_dir / "checkpoints" / f"step-{trainer.step_count:05d}")
        raise
    mo.output.replace(mo.vstack([mo.md(f"Done. Final adapter: `{trained_adapter}`"), plot_metrics(_history)]))
    return (trained_adapter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Evaluate base vs. trained

    Same eval problems, sampling seeds, student, and judge for both. The trained side uses `final/adapter` from the run directory, so this also works in a new session after a finished run.
    """)
    return


@app.cell
def _(mo):
    eval_button = mo.ui.run_button(label="Evaluate base vs. trained")
    eval_button
    return (eval_button,)


@app.cell
def _(cfg, eval_button, eval_problems, judge, mo, run_dir, student, tutor):
    mo.stop(not eval_button.value, mo.md("*Click to evaluate.*"))
    from tutor_rl.evaluation import evaluate
    from tutor_rl.policy import load_adapter_weights

    _adapter = run_dir / "final" / "adapter"
    assert _adapter.exists(), f"No trained adapter at {_adapter}; train first."
    load_adapter_weights(tutor, _adapter)

    def _switch(name, policy):
        policy.use_adapter = name == "trained"

    with mo.status.spinner(title="Evaluating…"):
        eval_report = evaluate(
            cfg, {"base": tutor, "trained": tutor}, eval_problems, student, judge,
            run_dir / "eval", before_each=_switch,
        )
    tutor.use_adapter = True
    mo.md((run_dir / "eval" / "results.md").read_text())
    return (eval_report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Export

    The zip holds what week 2 and the write-up need: `final/adapter` (LoRA weights and tokenizer), `config.toml`, `data/` (train/eval JSONL and the manifest with the data version), `metrics.jsonl`, every step's rollouts, `eval/` results and episodes, and `provenance.json` (repo commit, model and dataset revisions, package versions, GPU). Optimizer states and intermediate adapters stay out of the zip; they are only needed to resume.
    """)
    return


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Build export zip")
    export_button
    return (export_button,)


@app.cell
def _(cfg, data_manifest, export_button, mo, repo_commit, repo_dir, run_dir):
    mo.stop(not export_button.value)
    from tutor_rl.export import provenance, write_bundle

    _info = provenance(
        cfg,
        extra={
            "repo_commit": repo_commit,
            "data_version": data_manifest["data_version"],
            "notebook": "training/tutor_rl_molab.py",
            "steps_completed": len(read_metrics(run_dir)),
        },
    )
    export_zip = write_bundle(run_dir, _info)
    mo.vstack([
        mo.md(f"`{export_zip}` ({export_zip.stat().st_size / 1024**2:.0f} MB)"),
        mo.download(
            data=lambda: export_zip.read_bytes(),
            filename=export_zip.name,
            mimetype="application/zip",
            label="Download run bundle",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional: push to the Hugging Face Hub

    Uploads the run directory (adapter, checkpoints, logs, eval) to a **private** model repo. *Also push merged* additionally saves base + adapter as one bf16 text-only checkpoint (~18 GB) to `<repo>-merged`, which can be served without LoRA support. Stop the servers first if GPU memory is tight.
    """)
    return


@app.cell
def _(cfg, mo):
    hub_repo = mo.ui.text(placeholder=f"your-username/{cfg.run_name}", label="Hub repo id", full_width=True)
    push_merged = mo.ui.checkbox(False, label="Also push merged bf16 model")
    push_button = mo.ui.run_button(label="Push to Hub")
    mo.vstack([hub_repo, push_merged, push_button])
    return hub_repo, push_button, push_merged


@app.cell
def _(cfg, hub_repo, mo, push_button, push_merged, run_dir):
    mo.stop(not push_button.value)
    from tutor_rl.export import push_to_hub, save_merged

    assert hub_repo.value.strip(), "Enter a Hub repo id."
    _token = os.environ.get("HF_TOKEN")
    _urls = [push_to_hub(run_dir, hub_repo.value.strip(), token=_token)]
    if push_merged.value:
        with mo.status.spinner(title="Merging adapter into the base model…"):
            _merged = save_merged(run_dir / "final" / "adapter", cfg.tutor, run_dir / "merged")
        _urls.append(push_to_hub(_merged, hub_repo.value.strip() + "-merged", token=_token, ignore_patterns=[]))
    mo.md("Pushed: " + ", ".join(_urls))
    return


if __name__ == "__main__":
    app.run()
