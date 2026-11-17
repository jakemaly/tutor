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
    # It lets the tutor's batched generation and training reuse freed memory without fragmenting.
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
        "math-verify": "0.9.0",  # symbolic answer checking for Big-Math's LaTeX answers
    }


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutor RL: replicating PedagogicalRL with Qwen3.5-9B (issue #46)

    Replicates [*From Problem-Solving to Teaching Problem-Solving*](https://aclanthology.org/2025.emnlp-main.15) (Dinucu-Jianu et al., EMNLP 2025, CC BY 4.0) with the app's tutor model, **Qwen3.5-9B** (thinking off, bf16 LoRA), in place of the paper's Qwen2.5-7B. Everything else follows the paper:

    - **Data:** the paper's 10,000 train and 500 test Big-Math problems (`rd211/Big-Math-RL-Verified-Filtered`).
    - **Student:** Llama-3.1-8B-Instruct (FP8).
    - **Judges:** Qwen2.5-14B-Instruct-AWQ, using the leakage and helpfulness prompts, each sampled twice. Gemma-3-27B is the held-out leak judge for evaluation.
    - **Prompts:** the paper's, verbatim.
    - **Dialogues:** up to 16 messages. Either the student or the tutor opens.
    - **Step size:** 16 problems × 8 rollouts per step, μ = 2 gradient steps per batch, β = 0.001, and 150 steps (about 300 policy updates).
    - **Reward:** r = r_sol + (r_ped − 1)·λ + r_templ, with λ = 0.75.

    The code is in `training/tutor_rl/`, and every choice the paper doesn't pin down is listed in `training/replication/spec/assumptions_and_unknowns.md`. This notebook only wires it together.

    **Where things live:** GitHub holds this notebook, the `tutor_rl` code, and the configs. The **`socratictutor` Hugging Face org** holds everything a run produces. Each checkpoint (adapter + optimizer), the metrics, rollouts, logs (`logs/train.log` and the vLLM logs), and eval results upload to a private model repo as soon as they are saved. The problem set goes to a dataset repo. molab keeps no file over 1 GB after a shutdown and ends GPU sessions after 12 hours, so the Hub copy is the real one.

    **How long it takes:** the paper needed about 48 hours on 4× A100. Expect the full run to span several molab sessions; each one resumes from the newest Hub checkpoint. The smoke run reports seconds per step, so you can estimate the total before committing to it.

    **Before you start**

    1. Join the `socratictutor` org on Hugging Face and accept the [Llama 3.1 licence](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
    2. Create an access token (*Settings → Access Tokens*). Either a *Write* token, or a fine-grained token with read access to public gated repos plus write access to the `socratictutor` org. Paste it below.
    3. Turn on the GPU in molab's notebook specs. Expected: RTX PRO 6000, 96 GB.
    4. Run **smoke** first (a few minutes after downloads), then **molab**. If the session ends, open the notebook again, keep *Resume* ticked, and it continues from the newest checkpoint on the Hub. molab also stops notebooks that sit idle for 90 minutes, so keep the tab open while training.

    **GPU layout:** student vLLM ~19 GB, judge vLLM ~17 GB, tutor training ~55 GB. Evaluation adds the Gemma-3-27B leak judge (~25 GB). Each new session downloads about 60 GB of weights (a few minutes).
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
    hub_org = mo.ui.text(value="socratictutor", label="Hugging Face org", full_width=True)
    hub_run_name = mo.ui.text(
        placeholder="defaults to the config's run_name; use a new name for a new experiment",
        label="Hub run repo name",
        full_width=True,
    )
    hub_sync = mo.ui.checkbox(True, label="Upload checkpoints, logs and results to the Hub as they are saved")
    mo.vstack([
        config_choice, hf_token, repo_ref, work_root, max_steps_override, share_judge,
        hub_org, hub_run_name, hub_sync,
    ])
    return (
        config_choice,
        hf_token,
        hub_org,
        hub_run_name,
        hub_sync,
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
    | Judge | `{cfg.judge['model'] if not cfg.judge['share_student'] else cfg.student['model'] + ' (shared)'}` | `{cfg.judge['revision'][:12]}` |
    | Eval leak judge | `{cfg.eval_judge.get('model', '(training judge)')}` | `{cfg.eval_judge.get('revision', '')[:12]}` |

    {_t['max_steps']} steps × {_t['problems_per_step']} problems × {_t['episodes_per_problem']} rollouts
    ({_t['max_steps'] * _t.get('num_iterations', 1)} policy updates, μ = {_t.get('num_iterations', 1)}),
    up to {cfg.episode.max_turns} messages, K = {cfg.episode.student_final_attempts} final solutions,
    opener {cfg.episode.initiator}, λ = {cfg.reward.lam}{' (hard)' if cfg.reward.hard else ''},
    judges {', '.join(cfg.reward.judges)} × {cfg.reward.samples_per_judge},
    lr {_t['learning_rate']} ({_t.get('lr_scheduler', 'linear')}), β {_t['beta']},
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
    _sources = data_manifest["files"]["train.jsonl"].get("sources", {})
    mo.md(
        f"Data version `{data_manifest['data_version']}`: {len(train_problems)} train / "
        f"{len(eval_problems)} eval problems from `{data_manifest['dataset']}` "
        f"@ `{data_manifest['revision'][:12]}`.  \n"
        f"Train sources: {', '.join(f'{_k} {_v}' for _k, _v in _sources.items())} "
        f"(paper Table 1: big_math 3360, cn_k12 3324, math 1264, aops_forum 1263, omnimath 374, "
        f"openmath 315, gsm8k 100).  \n"
        f"Example: *{train_problems[0].question}* → `{train_problems[0].answer}`"
    )
    return data_manifest, eval_problems, train_problems


@app.cell
def _(cfg, data_manifest, hub_org, hub_run_name, hub_sync, mo, run_dir):
    # The run's private model repo (checkpoints, logs, results) and the shared dataset repo.
    from tutor_rl.hub import HubSync, push_dataset

    hub_repo_id = f"{hub_org.value.strip()}/{hub_run_name.value.strip() or cfg.run_name}"
    hub_dataset_id = f"{hub_org.value.strip()}/{cfg.data.get('hub_dataset', 'pedagogicalrl-bigmath')}"
    if hub_sync.value:
        _token = os.environ.get("HF_TOKEN")
        assert _token, "Paste a Hugging Face token above (or untick the Hub upload option)."
        hub = HubSync(hub_repo_id, token=_token)
        _data_url = push_dataset(run_dir / "data", hub_dataset_id, token=_token)
        _out = mo.md(
            f"Run repo: [{hub_repo_id}]({hub.url}) (private)  \n"
            f"Data version `{data_manifest['data_version']}`: [{hub_dataset_id}]({_data_url})"
        )
    else:
        hub = None
        _out = mo.callout(
            mo.md("Hub upload is off: everything stays in this molab session and is lost when it ends."),
            kind="warn",
        )
    _out
    return hub, hub_repo_id


@app.function
def gpu_report_md():
    """GPU memory in use, per process when nvidia-smi can tell (Markdown)."""
    from tutor_rl.vllm_server import gpu_report

    return gpu_report()


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
    from huggingface_hub import model_info

    from tutor_rl.vllm_server import ensure_vllm_env, start_student_and_judge
    from tutor_rl.vllm_server import served_models as _served_models

    # Start on click, but also re-adopt servers that are already running: this cell re-runs whenever
    # `cfg` changes (e.g. editing max_steps), and halting there would wipe `servers` for every cell
    # below it (the tutor load, training, …).
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

    # vLLM processes get a clean environment (no kernel PYTHONPATH/VIRTUAL_ENV, FlashInfer sampler
    # off, the venv first on PATH): see `vllm_env` in tutor_rl/vllm_server.py. A fresh start also
    # kills vLLM processes left over from an earlier failed start, which otherwise keep ~20 GB each.
    with mo.status.spinner(title="Installing vLLM (first run only)…"):
        _vllm_bin = ensure_vllm_env(Path(work_root.value) / "vllm-env")
    with mo.status.spinner(title="Starting student and judge (downloads weights on first run)…"):
        servers = start_student_and_judge(cfg, _vllm_bin, run_dir / "logs")
    mo.vstack([
        mo.md("Running: " + ", ".join(f"`{_s.name}` at {_s.base_url}" for _s in servers)),
        mo.md(gpu_report_md()),
    ])
    return (servers,)


@app.cell
def _(mo):
    stop_servers = mo.ui.run_button(label="Stop servers (frees ~36 GB)")
    show_gpu = mo.ui.run_button(label="Show GPU memory")
    mo.hstack([stop_servers, show_gpu], justify="start")
    return show_gpu, stop_servers


@app.cell
def _(mo, show_gpu, stop_servers):
    mo.stop(not (stop_servers.value or show_gpu.value))
    from tutor_rl.vllm_server import kill_stale_vllm

    _msg = ""
    if stop_servers.value:  # every vLLM process, including servers re-adopted from an earlier run
        _killed = kill_stale_vllm()
        _msg = f"Stopped {len(_killed)} vLLM process(es). Re-run the start cell to bring them back.\n\n"
    mo.md(_msg + gpu_report_md())
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
    from tutor_rl.openai_chat import model_from_section
    from tutor_rl.policy import load_tutor

    with mo.status.spinner(title=f"Loading {cfg.tutor['model']} (bf16 + LoRA)…"):
        tutor = load_tutor(cfg.tutor)
    student = model_from_section(cfg.student)
    # The pre- and post-dialog solutions are full step-by-step answers: a larger token budget.
    final_student = model_from_section(cfg.student, max_tokens_key="max_tokens_final")
    judge = model_from_section(cfg.judge)
    _trainable = sum(_p.numel() for _p in tutor.model.parameters() if _p.requires_grad)
    mo.md(f"Tutor loaded: {_trainable / 1e6:.1f}M trainable LoRA parameters.")
    return final_student, judge, student, tutor


@app.function
def rollout_view(r: dict, title: str):
    """One scored dialogue (a rollouts/*.jsonl or eval episodes row) as a collapsible entry."""
    import marimo as mo

    chat = "\n\n".join(
        f"**{'Teacher' if t['speaker'] == 'tutor' else 'Student'}:** {t['content']}"
        for t in r["turns"]
    )
    cut = [s.get("truncated", False) for s in r.get("tutor_steps", [])]
    judges = {
        f"{kind} judge, sample {i + 1}": mo.md(f"```\n{out}\n```")
        for kind, outs in r.get("judge_outputs", {}).items()
        for i, out in enumerate(outs)
    }
    for i, out in enumerate(r.get("eval_judge_outputs", [])):
        judges[f"eval leak judge, sample {i + 1}"] = mo.md(f"```\n{out}\n```")
    finals = "\n\n".join(f"{i + 1}. {a}" for i, a in enumerate(r["final_attempts"]))
    body = mo.vstack([
        mo.md(
            f"**Problem** ({r['problem'].get('source', '')}, answer `{r['problem']['answer']}`): "
            f"{r['problem']['question']}\n\n*Opened by the {r.get('initiator', 'student')}.*"
        ),
        mo.md(chat),
        mo.md(
            f"**Reward** {r['reward']:.2f} = r_sol {r['solve_rate']:.2f} "
            f"+ r_templ {r.get('template', 0):.2f} "
            f"{'' if r.get('accepted', True) else '− λ (judges rejected)'} · "
            f"rejected by: {r.get('rejected_by', {})} · ended by tutor {r['ended_by_tutor']} · "
            f"cut-off turns {sum(cut)}"
            + (f" · advantage {r['advantage']:.2f}" if "advantage" in r else "")
        ),
        mo.accordion({"Student's final solutions": mo.md(finals), **judges}),
    ])
    return (title, body)


@app.cell
def _(cfg, eval_problems, final_student, judge, mo, student, tutor):
    from tutor_rl.episode import run_episodes
    from tutor_rl.rewards import score_dict, score_episodes

    # One dialogue per opener with the base tutor: checks prompts, student, and both judges.
    tutor.use_adapter = False
    _eps = run_episodes(
        [eval_problems[0], eval_problems[0]], tutor, student, cfg.episode,
        initiators=["student", "tutor"], final_student=final_student,
    )
    _scores = score_episodes(_eps, judge, cfg.reward)
    tutor.use_adapter = True
    mo.vstack([
        mo.md("**Sanity check: base tutor, one dialogue per opener**"),
        mo.accordion(dict(
            rollout_view({**_e.to_dict(), **score_dict(_s)}, f"{_e.initiator}-initiated · reward {_s.reward:.2f}")
            for _e, _s in zip(_eps, _scores)
        )),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Train

    Each step appends to `metrics.jsonl`, `rollouts/step-*.jsonl` and `logs/train.log`, and refreshes the live view below: a status line, the newest `train.log` lines, and that step's dialogues. The full graphs and logs have their own sections after this one, and you can reopen them at any time, including in a new session. To stop early, interrupt the cell: the current step is saved as a checkpoint, and **Resume** picks it up later, also from the Hub after a new molab session.
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
def log_tail(path, lines=40):
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
        items = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            r = json.loads(line)
            if "accepted" not in r:  # written by the pre-replication trainer
                continue
            title = (
                f"Dialogue {i + 1} · {r['problem']['id']} · {r.get('initiator', '?')} opens · "
                f"reward {r['reward']:.2f} · solve {r['solve_rate']:.2f} · "
                f"{'accepted' if r['accepted'] else 'rejected'}"
            )
            items.append(rollout_view(r, title))
        return mo.vstack([
            mo.md(f"**Chat logs, step {step}** ({len(items)} dialogues; each group of "
                  "rollouts on the same problem is consecutive)"),
            mo.accordion(dict(items)),
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
    from tutor_rl.plots import training_dashboard

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
        _done = m["step"] - _start
        _eta = (time.time() - _t0) / _done * (_total - m["step"]) / 3600
        mo.output.replace(mo.vstack([
            mo.md(
                f"Step **{m['step']}/{_total}** ({m.get('policy_updates', '?')} policy updates) · "
                f"reward {m['reward']:.3f} · r_sol {m['solve_rate']:.2f} · accepted {m['accept_rate']:.2f} "
                f"(leakage rejects {m.get('leakage_reject_rate', 0):.2f}, helpfulness rejects "
                f"{m.get('helpfulness_reject_rate', 0):.2f}) · {m['step_seconds'] / 60:.1f} min/step "
                f"(dialogues {m['dialog_seconds'] / 60:.1f}, judges {m['judge_seconds'] / 60:.1f}) · "
                f"~{_eta:.1f} h left · peak GPU {m['max_gpu_gb']} GB"
            ),
            training_dashboard(_history),
            mo.accordion({"logs/train.log (last 15 lines)": mo.md(f"```\n{log_tail(run_dir / 'logs' / 'train.log', 15)}\n```")}),
            show_rollouts(run_dir, m["step"]),
        ]))

    def _upload(path):
        if hub is not None:
            hub.upload_checkpoint(run_dir, path)

    mo.output.replace(mo.md(f"Starting at step {_start}…"))
    try:
        trained_adapter = trainer.train(on_step=_show, on_save=_upload) / "adapter"
    except BaseException as _err:
        if "out of memory" in str(_err).lower():  # say where the memory went before re-raising
            from torch.cuda import memory_summary as _memory_summary

            mo.output.replace(mo.vstack([
                mo.callout(mo.md(f"**GPU out of memory** at step {trainer.step_count + 1}."), kind="danger"),
                mo.md(gpu_report_md()),
                mo.md(f"```\n{_memory_summary(abbreviated=True)}\n```"),
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
        training_dashboard(_history),
        show_rollouts(run_dir, trainer.step_count),
    ]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Training graphs and logs

    These read the run directory, so they work during training (click refresh) and in a later session. The **Fetch from Hub** button first downloads `metrics.jsonl`, the rollouts, and the logs of a run that trained elsewhere.

    How to read the graphs: **reward** and **r_sol** should rise, and **leakage rejects** should fall. **KL** should stay small (well under 0.1). Many **groups with no signal** means most problems are too easy or too hard for the student, so those steps teach little. A rising **cut-off turns** rate means the tutor is writing turns longer than `max_tokens_per_turn`.
    """)
    return


@app.cell
def _(mo):
    refresh_graphs = mo.ui.run_button(label="Refresh graphs and logs")
    fetch_hub_logs = mo.ui.run_button(label="Fetch metrics, rollouts and logs from the Hub")
    smooth_window = mo.ui.slider(start=1, stop=30, value=10, show_value=True, label="Moving average (steps)")
    mo.hstack([refresh_graphs, fetch_hub_logs, smooth_window], justify="start")
    return fetch_hub_logs, refresh_graphs, smooth_window


@app.cell
def _(fetch_hub_logs, hub_repo_id, mo, refresh_graphs, run_dir, smooth_window):
    refresh_graphs.value  # re-run on click
    from tutor_rl.plots import training_dashboard as _dashboard

    if fetch_hub_logs.value:
        from huggingface_hub import snapshot_download as _snapshot

        with mo.status.spinner(title=f"Downloading logs from {hub_repo_id}…"):
            _snapshot(
                hub_repo_id, local_dir=str(run_dir), token=os.environ.get("HF_TOKEN"),
                allow_patterns=["metrics.jsonl", "rollouts/**", "logs/**"],
            )
    _rows = read_metrics(run_dir)
    _logs = {
        _p.name: mo.md(f"```\n{log_tail(_p, 200)}\n```")
        for _p in sorted((run_dir / "logs").glob("*.log"))
    } if (run_dir / "logs").exists() else {}
    mo.vstack([
        mo.md(f"`{run_dir / 'metrics.jsonl'}`: {len(_rows)} steps logged."),
        _dashboard(_rows, window=smooth_window.value),
        mo.accordion({"GPU memory now": mo.md(gpu_report_md())}),
        mo.md("**Logs** (last 200 lines each: the trainer's `train.log` and each vLLM server)"),
        mo.ui.tabs(_logs) if _logs else mo.md("_No logs yet._"),
    ])
    return


@app.cell
def _(mo, refresh_graphs, run_dir):
    refresh_graphs.value
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
    ## 6. Evaluate base vs. trained (the paper's Table 2)

    The paper's in-domain evaluation runs on the 500 held-out Big-Math problems, with the same openers, sampling seeds, student and judges for both tutors. It reports:

    - **Δ solve rate:** the student's post-dialog solve rate minus its pre-dialog chain-of-thought solve rate, in percentage points.
    - **Leaked solutions:** the share of dialogues the **Gemma-3-27B** leak judge rejects. The paper uses a judge from another model family at test time, so the tutor is never scored by the judge it trained against.

    The trained side uses `final/adapter`, so this also works in a new session after a finished run. With 500 dialogues per tutor, expect roughly an hour.
    """)
    return


@app.cell
def _(mo):
    use_eval_judge = mo.ui.checkbox(True, label="Start the Gemma-3-27B leak judge (paper's test-time judge, ~25 GB)")
    eval_button = mo.ui.run_button(label="Evaluate base vs. trained")
    mo.hstack([eval_button, use_eval_judge], justify="start")
    return eval_button, use_eval_judge


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
    use_eval_judge,
    work_root,
):
    mo.stop(not eval_button.value, mo.md("*Click to evaluate.*"))
    import gc

    import torch as _torch

    from tutor_rl.evaluation import evaluate
    from tutor_rl.openai_chat import model_from_section as _from_section
    from tutor_rl.policy import load_adapter_weights
    from tutor_rl.vllm_server import ensure_vllm_env as _ensure_vllm
    from tutor_rl.vllm_server import start_eval_judge

    _adapter = run_dir / "final" / "adapter"
    if not _adapter.exists() and hub is not None:  # trained in an earlier session
        from huggingface_hub import snapshot_download

        snapshot_download(
            hub_repo_id, local_dir=str(run_dir), allow_patterns=["final/**"],
            token=os.environ.get("HF_TOKEN"),
        )
    assert _adapter.exists(), f"No trained adapter at {_adapter}; train first."
    load_adapter_weights(tutor, _adapter)

    _eval_judge = None
    if use_eval_judge.value and cfg.eval_judge:
        gc.collect()
        _torch.cuda.empty_cache()  # give the training run's cached blocks back for vLLM
        with mo.status.spinner(title=f"Starting {cfg.eval_judge['model']} (downloads ~20 GB once)…"):
            start_eval_judge(cfg, _ensure_vllm(Path(work_root.value) / "vllm-env"), run_dir / "logs")
        _eval_judge = _from_section(cfg.eval_judge)

    def _switch(name, policy):
        policy.use_adapter = name == "trained"

    with mo.status.progress_bar(total=5, title="Evaluating…", remove_on_exit=True) as _bar:
        evaluate(
            cfg, {"base": tutor, "trained": tutor}, eval_problems, student, judge,
            run_dir / "eval", before_each=_switch, eval_judge=_eval_judge,
            final_student=final_student,
            on_progress=lambda _msg: _bar.update(title=_msg),
        )
    tutor.use_adapter = True
    if hub is not None:
        hub.upload_run(run_dir, include=["eval/**", "logs/**"], message="Evaluation").result()
    mo.md(f"Evaluation written to `{run_dir / 'eval'}`" + (" and uploaded." if hub else "."))
    return


@app.cell
def _(eval_button, fetch_hub_logs, hub_repo_id, mo, run_dir):
    eval_button.value, fetch_hub_logs.value  # refresh after an evaluation or a Hub fetch
    from tutor_rl.plots import eval_bars, eval_tradeoff

    _results = run_dir / "eval" / "results.json"
    if not _results.exists() and os.environ.get("HF_TOKEN"):
        try:  # evaluated in an earlier session
            from huggingface_hub import snapshot_download as _snap

            _snap(hub_repo_id, local_dir=str(run_dir), allow_patterns=["eval/**"], token=os.environ["HF_TOKEN"])
        except Exception:  # noqa: BLE001 - no run repo yet
            pass
    mo.stop(not _results.exists(), mo.md("_No evaluation results yet._"))
    eval_report = json.loads(_results.read_text())
    mo.vstack([
        eval_tradeoff(eval_report),
        eval_bars(eval_report),
        mo.md((run_dir / "eval" / "results.md").read_text(encoding="utf-8")),
    ])
    return (eval_report,)


@app.cell
def _(eval_report, mo):
    eval_policy = mo.ui.dropdown(list(eval_report["results"]), value=list(eval_report["results"])[-1], label="Eval dialogues for")
    eval_only = mo.ui.dropdown(["all", "leaked (eval judge)", "not leaked"], value="all", label="Show")
    mo.hstack([eval_policy, eval_only], justify="start")
    return eval_only, eval_policy


@app.cell
def _(eval_only, eval_policy, mo, run_dir):
    _rows = [
        json.loads(_l)
        for _l in (run_dir / "eval" / f"episodes-{eval_policy.value}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if eval_only.value != "all":
        _rows = [_r for _r in _rows if _r["eval_leaked"] == (eval_only.value.startswith("leaked"))]
    mo.vstack([
        mo.md(f"{len(_rows)} dialogues (first 50 shown)"),
        mo.accordion(dict(
            rollout_view(_r, f"{_r['problem']['id']} · solve {_r['solve_rate']:.2f} · "
                             f"{'LEAKED' if _r['eval_leaked'] else 'no leak'}")
            for _r in _rows[:50]
        )),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Export

    Writes `provenance.json` (repo commit, model and dataset revisions, package versions, GPU) and uploads it to the run repo. With Hub upload on, the run repo then holds everything: checkpoints, `final/adapter`, metrics, rollouts, logs, eval, and provenance.

    The zip is an optional local copy of the same files, minus optimizer states and intermediate adapters. Download it before the session ends; molab will not keep it.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional: merged model for serving

    Saves base + adapter as one bf16 text-only checkpoint (~18 GB) and uploads it to `<run repo>-merged` (private), for serving without LoRA support in week 2. Stop the servers first if GPU memory is tight.
    """)
    return


@app.cell
def _(mo):
    merge_button = mo.ui.run_button(label="Merge and upload")
    merge_button
    return (merge_button,)


@app.cell
def _(cfg, hub_repo_id, merge_button, mo, run_dir):
    mo.stop(not merge_button.value)
    from tutor_rl.export import push_to_hub, save_merged

    with mo.status.spinner(title="Merging adapter into the base model…"):
        _merged = save_merged(run_dir / "final" / "adapter", cfg.tutor, run_dir / "merged")
    with mo.status.spinner(title="Uploading ~18 GB…"):
        _url = push_to_hub(
            _merged, hub_repo_id + "-merged", token=os.environ.get("HF_TOKEN"), ignore_patterns=[]
        )
    mo.md(f"Merged model: {_url}")
    return


if __name__ == "__main__":
    app.run()
