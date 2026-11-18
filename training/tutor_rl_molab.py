# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub==2.4.0",
#     "marimo",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])

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

    # Must be set before torch first touches CUDA (the dependency cell does), or it has no effect.
    # The vLLM servers do not inherit it (tutor_rl.vllm_server.vllm_env drops it).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    REPO = "jakemaly/tutor"
    DEFAULT_REF = "46-ml-2-1-implement-the-pedagogicalrl-inspired-trainer-and-serve-its-checkpoint"

    # Training stack. torch comes from molab's GPU image; vLLM gets its own venv (see below).
    PINS = {
        "transformers": "5.23.0",
        "peft": "0.21.3",
        "accelerate": "1.15.0",
        "datasets": "5.2.0",
        "flash-linear-attention": "0.6.0",  # fast Gated DeltaNet kernels for Qwen3.5
        "math-verify": "0.9.0",  # checks the student's LaTeX answers
    }

    UNTRAINED, TRAINED = "#eb6834", "#2a78d6"  # chart colours: orange = untrained, blue = trained


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutor RL: teaching Qwen3.5-9B not to give answers away (issue #46)

    Trains the app's tutor model (**Qwen3.5-9B**, thinking off, bf16 LoRA) with multi-turn GRPO, following [PedagogicalRL](https://aclanthology.org/2025.emnlp-main.15) (EMNLP 2025):

    - **Student:** a simulated student, Llama-3.1-8B-Instruct.
    - **Problems:** Big-Math problems the student usually gets wrong.
    - **Judge:** Qwen2.5-14B-Instruct-AWQ checks every dialogue for **answer leaks** and for unhelpful teaching.
    - **Reward:** how often the student solves the problem afterwards, or **−1** if a judge rejects the dialogue.

    The goal is a clear before/after: the trained tutor should **leak the answer less** while the **student still solves as often or more**.

    **Before you start**

    1. Join the `socratictutor` org on Hugging Face and accept the [Llama 3.1 licence](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
    2. Create a *Write* access token (or a fine-grained one with read access to gated repos and write access to `socratictutor`). Paste it below.
    3. Turn on the GPU in molab's notebook specs (RTX PRO 6000, 96 GB).
    4. Run **smoke** first (a few minutes after downloads), then **molab** (60 steps × 24 dialogues). If the session ends, reopen the notebook with *Resume* ticked: it continues from the newest checkpoint on the Hub.

    Checkpoints, metrics, chat logs and results upload to a private repo in the `socratictutor` org as they are saved. GPU memory stays under 80%: student 15%, judge 15%, and the tutor capped at 50%.
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
    hub_org = mo.ui.text(value="socratictutor", label="Hugging Face org", full_width=True)
    hub_run_name = mo.ui.text(
        placeholder="defaults to the config's run_name; use a new name for a new experiment",
        label="Hub run repo name",
        full_width=True,
    )
    hub_sync = mo.ui.checkbox(True, label="Upload checkpoints, logs and results to the Hub as they are saved")
    mo.vstack([
        config_choice, hf_token, repo_ref, work_root, max_steps_override, hub_org, hub_run_name, hub_sync,
    ])
    return (
        config_choice,
        hf_token,
        hub_org,
        hub_run_name,
        hub_sync,
        max_steps_override,
        repo_ref,
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
    return repo_commit, training_dir


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

    # The molab image pairs the kernel's torch with a system torchvision built for a different torch,
    # so ``import torchvision`` raises "operator torchvision::nms does not exist". transformers imports
    # torchvision eagerly (reached through peft), which would break every model-loading cell. This
    # notebook is text-only, so hide torchvision and drop transformers' cached availability check.
    sys.modules["torchvision"] = None
    import transformers.utils.import_utils as _tf_utils

    for _tf_flag in ("is_torchvision_available", "is_torchvision_v2_available"):
        _tf_clear = getattr(getattr(_tf_utils, _tf_flag, None), "cache_clear", None)
        if _tf_clear is not None:
            _tf_clear()

    import torch

    assert torch.cuda.is_available(), "No GPU: turn on the GPU in molab's notebook specs."
    _p = torch.cuda.get_device_properties(0)
    mo.md(
        f"GPU: **{_p.name}**, {_p.total_memory / 1024**3:.0f} GB, CUDA {torch.version.cuda}  \n"
        f"Packages: `{_versions}`"
    )
    return (deps_ready,)


@app.cell
def _(config_choice, deps_ready, hf_token, max_steps_override, mo, training_dir, work_root):
    assert deps_ready
    from tutor_rl.config import load_config
    from tutor_rl.gpu_budget import plan as _gpu_plan

    _base = load_config(training_dir / "configs" / f"{config_choice.value}.toml")
    _train = dict(_base.train)
    if max_steps_override.value:
        _train["max_steps"] = int(max_steps_override.value)
    cfg = replace(_base, train=_train)
    run_dir = Path(work_root.value) / "outputs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if hf_token.value:
        os.environ["HF_TOKEN"] = hf_token.value

    _t = cfg.train
    mo.md(
        f"**Run** `{cfg.run_name}` → `{run_dir}`  \n"
        f"{_t['max_steps']} steps × {_t['problems_per_step']} problems × {_t['episodes_per_problem']} "
        f"dialogues ({_t['problems_per_step'] * _t['episodes_per_problem']} per step), up to "
        f"{cfg.episode.max_turns} messages each, lr {_t['learning_rate']}, checkpoint every "
        f"{_t['save_steps']} steps.  \nGPU budget: {_gpu_plan(cfg).summary()}."
    )
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
        f"{len(eval_problems)} eval problems from `{data_manifest['dataset']}`. "
        f"Example: *{train_problems[0].question}* → `{train_problems[0].answer}`"
    )
    return data_manifest, eval_problems, train_problems


@app.cell
def _(cfg, data_manifest, hub_org, hub_run_name, hub_sync, mo, run_dir):
    # The run's private model repo (checkpoints, logs, results) and the shared dataset repo.
    from tutor_rl.hub import HubSync, push_dataset

    hub_repo_id = f"{hub_org.value.strip()}/{hub_run_name.value.strip() or cfg.run_name}"
    _dataset_id = f"{hub_org.value.strip()}/{cfg.data.get('hub_dataset', 'tutor-rl-bigmath')}"
    if hub_sync.value:
        _token = os.environ.get("HF_TOKEN")
        assert _token, "Paste a Hugging Face token above (or untick the Hub upload option)."
        hub = HubSync(hub_repo_id, token=_token)
        _data_url = push_dataset(run_dir / "data", _dataset_id, token=_token)
        _out = mo.md(
            f"Run repo: [{hub_repo_id}]({hub.url}) (private)  \n"
            f"Data version `{data_manifest['data_version']}`: [{_dataset_id}]({_data_url})"
        )
    else:
        hub = None
        _out = mo.callout(
            mo.md("Hub upload is off: everything stays in this molab session and is lost when it ends."),
            kind="warn",
        )
    _out
    return hub, hub_repo_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Student and judge servers

    vLLM is installed once into its own venv (a few minutes), then the student and judge start as local servers. They start before the tutor loads, because each reserves a fixed share of GPU memory. Re-running this cell reuses servers that are already up; a fresh start first clears vLLM processes left over from a failed attempt.
    """)
    return


@app.cell
def _(mo):
    start_servers = mo.ui.run_button(label="Start student and judge")
    stop_servers = mo.ui.run_button(label="Stop servers")
    mo.hstack([start_servers, stop_servers], justify="start")
    return start_servers, stop_servers


@app.cell
def _(cfg, mo, run_dir, start_servers, work_root):
    from huggingface_hub import model_info

    from tutor_rl.vllm_server import ensure_vllm_env, gpu_report, start_student_and_judge
    from tutor_rl.vllm_server import served_models as _served_models

    # Start on click, but also re-adopt servers that are already running: this cell re-runs whenever
    # `cfg` changes, and halting there would wipe `servers` for every cell below it.
    _ports = [cfg.student["port"]] + ([] if cfg.judge.get("share_student") else [cfg.judge["port"]])
    _already_up = all(_served_models(f"http://127.0.0.1:{_p}/v1") for _p in _ports)
    mo.stop(not (start_servers.value or _already_up), mo.md("*Click to start the servers.*"))

    if not _already_up:  # fail fast on the gated Llama repo instead of deep inside vLLM's logs
        try:
            model_info(cfg.student["model"], revision=cfg.student["revision"], token=os.environ.get("HF_TOKEN"))
        except Exception as _e:
            raise RuntimeError(
                f"Cannot access {cfg.student['model']}: accept its licence on Hugging Face and "
                f"paste a token above. ({type(_e).__name__})"
            ) from None

    # vLLM gets a clean environment (no kernel PYTHONPATH, FlashInfer sampler off): see vllm_env.
    with mo.status.spinner(title="Installing vLLM (first run only)…"):
        _vllm_bin = ensure_vllm_env(Path(work_root.value) / "vllm-env")
    with mo.status.spinner(title="Starting student and judge (downloads weights on first run)…"):
        servers = start_student_and_judge(cfg, _vllm_bin, run_dir / "logs")
    mo.vstack([
        mo.md("Running: " + ", ".join(f"`{_s.name}` at {_s.base_url}" for _s in servers)),
        mo.md(gpu_report()),
    ])
    return (servers,)


@app.cell
def _(mo, stop_servers):
    mo.stop(not stop_servers.value)
    from tutor_rl.vllm_server import kill_stale_vllm

    mo.md(f"Stopped {len(kill_stale_vllm())} vLLM process(es). Click *Start* to bring them back.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Tutor and a sanity-check dialogue

    One dialogue per opener (student first, tutor first) with the **untrained** tutor, to check the prompts, the student and the judge before training.
    """)
    return


@app.cell
def _(cfg, deps_ready, mo, servers):
    assert deps_ready and servers  # GPU memory for vLLM is reserved first
    from tutor_rl.gpu_budget import cap_tutor, plan
    from tutor_rl.openai_chat import model_from_section
    from tutor_rl.policy import load_tutor

    cap_tutor(plan(cfg))  # hard cap: what the servers leave under 80% of the GPU
    with mo.status.spinner(title=f"Loading {cfg.tutor['model']} (bf16 + LoRA)…"):
        tutor = load_tutor(cfg.tutor)
    student = model_from_section(cfg.student)
    # The student's solutions before/after the dialogue are full step-by-step answers.
    final_student = model_from_section(cfg.student, max_tokens_key="max_tokens_final")
    judge = model_from_section(cfg.judge)
    _trainable = sum(_p.numel() for _p in tutor.model.parameters() if _p.requires_grad)
    mo.md(f"Tutor loaded: {_trainable / 1e6:.1f}M trainable LoRA parameters.")
    return final_student, judge, student, tutor


@app.function
def dialogue_view(r: dict, title: str):
    """One scored dialogue (a rollouts/*.jsonl or eval row) as a collapsible entry."""
    import marimo as mo

    chat = "\n\n".join(
        f"**{'Tutor' if t['speaker'] == 'tutor' else 'Student'}:** {t['content']}" for t in r["turns"]
    )
    judge = "\n\n".join(
        f"*{kind}:* {outs[0]}" for kind, outs in r.get("judge_outputs", {}).items() if outs
    )
    body = (
        f"**Problem** (answer `{r['problem']['answer']}`): {r['problem']['question']}\n\n{chat}\n\n"
        f"**Student solved afterwards:** {r['solve_rate']:.0%} of attempts · "
        f"**leaked:** {r.get('leaked', '?')} · **reward** {r['reward']:.2f}\n\n"
        f"**Judge:**\n\n{judge}"
    )
    return (title, mo.md(body))


@app.cell
def _(cfg, eval_problems, final_student, judge, mo, student, tutor):
    from tutor_rl.episode import run_episodes
    from tutor_rl.rewards import score_dict, score_episodes

    tutor.use_adapter = False
    _eps = run_episodes(
        [eval_problems[0], eval_problems[0]], tutor, student, cfg.episode,
        initiators=["student", "tutor"], final_student=final_student,
    )
    _scores = score_episodes(_eps, judge, cfg.reward)
    tutor.use_adapter = True
    mo.accordion(dict(
        dialogue_view({**_e.to_dict(), **score_dict(_s)}, f"{_e.initiator} opens · reward {_s.reward:.2f}")
        for _e, _s in zip(_eps, _scores)
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Train

    Each step logs to `metrics.jsonl`, `rollouts/step-*.jsonl` and `logs/train.log`, and updates the plot below. **Answer leaked** should fall and **student solved** should hold or rise. To stop early, interrupt the cell: the current step is saved as a checkpoint, and **Resume** picks it up later.
    """)
    return


@app.cell
def _(mo):
    resume = mo.ui.checkbox(True, label="Resume from the latest checkpoint if there is one")
    train_button = mo.ui.run_button(label="Train")
    mo.hstack([train_button, resume], justify="start")
    return resume, train_button


@app.function
def read_metrics(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@app.function
def plot_training(rows, window=5):
    """Three curves: reward, answer-leak rate, student solve rate (dots per step, moving average)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    steps = [r["step"] for r in rows]
    panels = [
        ("reward", "Reward", 1),
        ("leak_rate", "Answer leaked (%)", 100),
        ("solve_rate", "Student solved after dialogue (%)", 100),
    ]
    for ax, (key, title, scale) in zip(axes, panels):
        values = [scale * r.get(key, float("nan")) for r in rows]
        smooth = [
            sum(values[max(0, i - window + 1) : i + 1]) / len(values[max(0, i - window + 1) : i + 1])
            for i in range(len(values))
        ]
        ax.scatter(steps, values, s=10, color=TRAINED, alpha=0.3)
        ax.plot(steps, smooth, color=TRAINED, linewidth=2)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


@app.function
def log_tail(path, lines=15):
    path = Path(path)
    if not path.exists():
        return f"({path.name} does not exist yet)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


@app.cell
def _(mo):
    def show_rollouts(run_dir, step):
        path = Path(run_dir) / "rollouts" / f"step-{step:05d}.jsonl"
        if not path.exists():
            return mo.md(f"_No chat logs for step {step} yet._")
        rows = [json.loads(_l) for _l in path.read_text(encoding="utf-8").splitlines()]
        rows = [_r for _r in rows if "accepted" in _r]  # skip logs from older trainer versions
        return mo.vstack([
            mo.md(f"**Chat logs, step {step}** ({len(rows)} dialogues)"),
            mo.accordion(dict(
                dialogue_view(
                    _r,
                    f"{i + 1} · {_r['problem']['id']} · reward {_r['reward']:.2f} · "
                    f"{'LEAKED' if _r.get('leaked') else 'no leak'} · solved {_r['solve_rate']:.0%}",
                )
                for i, _r in enumerate(rows)
            )),
        ])

    return (show_rollouts,)


@app.cell
def _(
    cfg,
    final_student,
    hub,
    hub_repo_id,
    judge,
    mo,
    resume,
    run_dir,
    show_rollouts,
    student,
    train_button,
    train_problems,
    tutor,
):
    mo.stop(not train_button.value, mo.md("*Click **Train** to start.*"))
    from tutor_rl.grpo import GRPOTrainer
    from tutor_rl.hub import fetch_for_resume

    if resume.value and hub is not None:  # a new molab session: pull the newest Hub checkpoint
        with mo.status.spinner(title="Looking for a checkpoint on the Hub…"):
            fetch_for_resume(hub_repo_id, run_dir, token=os.environ.get("HF_TOKEN"))

    if not resume.value:  # fresh run: set the previous attempt aside instead of mixing them
        _stamp = int(time.time())
        for _name in ("metrics.jsonl", "rollouts", "checkpoints", "final", "logs/train.log"):
            if (run_dir / _name).exists():
                (run_dir / _name).rename(run_dir / f"previous-{_stamp}-{_name.replace('/', '-')}")
    trainer = GRPOTrainer(
        cfg, tutor, student, judge, train_problems, run_dir, final_student=final_student
    )
    _start = trainer.resume() if resume.value else 0
    _history = [_r for _r in read_metrics(run_dir) if _r["step"] <= _start]
    _total = cfg.train["max_steps"]
    _t0 = time.time()

    def _show(m):
        _history.append(m)
        _eta = (time.time() - _t0) / (m["step"] - _start) * (_total - m["step"]) / 3600
        mo.output.replace(mo.vstack([
            mo.md(
                f"Step **{m['step']}/{_total}** · reward {m['reward']:.2f} · answer leaked "
                f"{100 * m['leak_rate']:.0f}% · student solved {100 * m['solve_rate']:.0f}% · "
                f"{m['step_seconds'] / 60:.1f} min/step · ~{_eta:.1f} h left · peak GPU {m['max_gpu_gb']} GB"
            ),
            plot_training(_history),
            mo.accordion({"logs/train.log": mo.md(f"```\n{log_tail(run_dir / 'logs' / 'train.log')}\n```")}),
            show_rollouts(run_dir, m["step"]),
        ]))

    def _upload(path):
        if hub is not None:
            hub.upload_checkpoint(run_dir, path)

    mo.output.replace(mo.md(f"Starting at step {_start}…"))
    try:
        trained_adapter = trainer.train(on_step=_show, on_save=_upload) / "adapter"
    except BaseException as _err:
        if "out of memory" in str(_err).lower():  # show where the memory went before re-raising
            from tutor_rl.vllm_server import gpu_report as _gpu_report

            mo.output.replace(mo.vstack([
                mo.callout(mo.md(f"**GPU out of memory** at step {trainer.step_count + 1}."), kind="danger"),
                mo.md(_gpu_report()),
            ]))
        if trainer.step_count > _start:  # keep the progress made since the last checkpoint
            _upload(trainer.save(trainer.checkpoint_path()))
            if hub is not None:
                hub.wait()
        raise
    if hub is not None:
        with mo.status.spinner(title="Finishing Hub uploads…"):
            hub.wait()
    _where = f" and [on the Hub]({hub.url})" if hub is not None else ""
    mo.output.replace(mo.vstack([
        mo.md(f"Done. Final adapter: `{trained_adapter}`{_where}."),
        plot_training(_history),
    ]))
    return


@app.cell
def _(mo, run_dir, train_button):
    train_button.value  # refresh after training
    _n = len(list((run_dir / "rollouts").glob("step-*.jsonl")))
    log_step = mo.ui.slider(start=1, stop=max(_n, 1), value=max(_n, 1), show_value=True, label="Chat logs for step")
    log_step
    return (log_step,)


@app.cell
def _(log_step, run_dir, show_rollouts):
    show_rollouts(run_dir, log_step.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Untrained vs. trained

    Both tutors tutor the same held-out problems with the same openers, sampling seeds, student and judge. The trained side uses `final/adapter`, so this also works in a new session after a finished run.
    """)
    return


@app.cell
def _(mo):
    eval_button = mo.ui.run_button(label="Evaluate untrained vs. trained")
    eval_button
    return (eval_button,)


@app.function
def plot_comparison(report):
    """Grouped bars: answer leaked and student solved, untrained vs. trained."""
    import matplotlib.pyplot as plt

    names = list(report["results"])
    metrics = [("leak_rate", "Answer leaked (%)\nlower is better"), ("post_solve_rate", "Student solved (%)\nhigher is better")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        values = [report["results"][name][k] for k, _ in metrics]
        xs = [i + (j - (len(names) - 1) / 2) * width for i in range(len(metrics))]
        bars = ax.bar(xs, values, width * 0.92, color=UNTRAINED if j == 0 else TRAINED, label=name)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:.0f}%", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9)
    pre = next(iter(report["results"].values()))["pre_solve_rate"]
    ax.plot([0.55, 1.45], [pre, pre], color="#6b6b66", linestyle="--", linewidth=1,
            label=f"student alone, before any dialogue ({pre:.0f}%)")
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


@app.cell
def _(
    cfg,
    eval_button,
    eval_problems,
    final_student,
    hub,
    hub_repo_id,
    judge,
    mo,
    run_dir,
    student,
    tutor,
):
    mo.stop(not eval_button.value, mo.md("*Click to evaluate.*"))
    from tutor_rl.evaluation import evaluate
    from tutor_rl.policy import load_adapter_weights

    _adapter = run_dir / "final" / "adapter"
    if not _adapter.exists() and hub is not None:  # trained in an earlier session
        from huggingface_hub import snapshot_download

        snapshot_download(
            hub_repo_id, local_dir=str(run_dir), allow_patterns=["final/**"],
            token=os.environ.get("HF_TOKEN"),
        )
    assert _adapter.exists(), f"No trained adapter at {_adapter}; train first."
    load_adapter_weights(tutor, _adapter)

    def _switch(name, policy):
        policy.use_adapter = name == "trained"

    with mo.status.spinner(title="Evaluating both tutors on the same scenarios…"):
        eval_report = evaluate(
            cfg, {"untrained": tutor, "trained": tutor}, eval_problems, student, judge,
            run_dir / "eval", before_each=_switch, final_student=final_student,
        )
    tutor.use_adapter = True
    if hub is not None:
        hub.upload_run(run_dir, include=["eval/**"], message="Evaluation").result()
    mo.vstack([
        plot_comparison(eval_report),
        mo.md((run_dir / "eval" / "results.md").read_text(encoding="utf-8")),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Export

    Writes `provenance.json` (repo commit, model and dataset revisions, package versions, GPU) and uploads it to the run repo, plus an optional local zip of the run (no optimizer states). Download the zip before the session ends; molab will not keep it.
    """)
    return


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Build export zip")
    export_button
    return (export_button,)


@app.cell
def _(cfg, data_manifest, export_button, hub, mo, repo_commit, run_dir):
    mo.stop(not export_button.value)
    from tutor_rl.export import provenance, write_bundle

    _info = provenance(
        cfg,
        extra={
            "repo_commit": repo_commit,
            "data_version": data_manifest["data_version"],
            "notebook": "training/tutor_rl_molab.py",
            "hub_repo": hub.repo_id if hub is not None else None,
            "steps_completed": len(read_metrics(run_dir)),
        },
    )
    export_zip = write_bundle(run_dir, _info)
    if hub is not None:
        hub.upload_run(run_dir, include=["provenance.json"], message="Provenance").result()
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


if __name__ == "__main__":
    app.run()
