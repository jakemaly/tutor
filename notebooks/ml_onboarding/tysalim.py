# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "datasets",
#     "hf_transfer",
#     "huggingface_hub",
#     "marimo",
#     "matplotlib",
#     "transformers==4.57.6",
#     "trl==0.22.2",
#     "unsloth",
#     "vllm",
# ]
# ///
"""Marimo recreation of Unsloth's pinned Qwen2.5-3B GRPO notebook for issue #21.

Upstream: https://github.com/unslothai/notebooks/blob/c55d0eccc722a7754036a447bdd5cd83499d4350/nb/Qwen2.5_(3B)-GRPO.ipynb

Training recipe (model, GSM8K prep, rewards, GRPOConfig including 250 steps)
is copied from that revision. Install cells and Colab magics are adapted for
Marimo / Molab. Weight-export cells are omitted so adapters are not committed.
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App()

@app.cell
def _():
    import marimo as mo

    UPSTREAM_NOTEBOOK_COMMIT = "c55d0eccc722a7754036a447bdd5cd83499d4350"
    UPSTREAM_NOTEBOOK_URL = (
        "https://github.com/unslothai/notebooks/blob/"
        f"{UPSTREAM_NOTEBOOK_COMMIT}/nb/Qwen2.5_(3B)-GRPO.ipynb"
    )
    MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
    return MODEL_NAME, UPSTREAM_NOTEBOOK_COMMIT, UPSTREAM_NOTEBOOK_URL, mo


@app.cell(hide_code=True)
def _(UPSTREAM_NOTEBOOK_COMMIT, UPSTREAM_NOTEBOOK_URL, mo):
    mo.md(
        rf"""
        # Qwen2.5-3B GRPO on GSM8K (Molab)

        Marimo recreation of Unsloth's official Qwen2.5-3B GRPO notebook for
        [tutor#21](https://github.com/jakemaly/tutor/issues/21).

        **Pinned source:** [`{UPSTREAM_NOTEBOOK_COMMIT}`]({UPSTREAM_NOTEBOOK_URL})

        This notebook keeps the upstream model, GSM8K preparation, reward
        functions, and GRPO hyperparameters (including **250 steps**). Colab
        magics and Hub/GGUF export are the only intentional changes.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What you still have to do in Molab

        This file is the repository-backed notebook. It does **not** contain a
        completed 250-step run until you execute it on a GPU.

        1. Push this branch and open the notebook on
           [Molab](https://molab.marimo.io) as a GitHub-synced notebook
           ([guide](https://docs.marimo.io/guides/molab/)).
        2. Attach a free **NVIDIA RTX Pro 6000 Blackwell** in notebook specs.
        3. **Run all**. Stay active: Molab stops idle sessions after 90 minutes.
           Unsloth warns that reward may stay near 0 for ~100 steps.
        4. Confirm the provenance table filled, the reward plot rose after the
           early plateau, and before/after generations rendered.
        5. Commit `tysalim_run_metrics.json` if that file was written next to
           this notebook. Do **not** commit `outputs/`, `grpo_saved_lora/`,
           `.safetensors`, or GGUF files.
        6. Open a PR from `ml-onboarding-TylerLim` into `dev`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## How to read this notebook

        The Hugging Face LLM course treats modern LLMs as **decoder-only
        Transformers**: they map token IDs to a next-token distribution
        (the policy). The Deep RL course treats learning as an agent that
        takes actions to maximize expected return. Unsloth's GRPO guide
        specializes that picture: sample a **group** of completions, score
        them with **verifiable** reward functions, and update without a
        learned critic or reward model.

        This notebook is a single-turn **solver**. PedagogicalRL uses GRPO
        too, but the episode is a classroom dialog whose reward **withholds**
        answers. The sections below are written from those sources, not from
        a full pass through either course.
        """
    )
    return


@app.cell
def _():
    import importlib.metadata as metadata
    import json
    import os
    import platform
    import subprocess
    import sys
    from pathlib import Path

    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Match the pinned notebook's non-Colab + version pins (transformers/trl).
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "unsloth",
            "vllm",
            "datasets",
            "matplotlib",
            "huggingface_hub",
        ]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6"]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "trl==0.22.2"]
    )

    artifact_dir = Path.cwd()
    return artifact_dir, json, metadata, platform, subprocess


@app.cell
def _(mo, subprocess):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. In Molab, open notebook specs, attach an "
            "NVIDIA RTX Pro 6000 Blackwell, then Run all again."
        )

    gpu_name = torch.cuda.get_device_name(0)
    nvidia_smi = subprocess.getoutput("nvidia-smi")
    gpu_status = mo.md(f"**GPU:** `{gpu_name}`")
    gpu_status
    return gpu_name, nvidia_smi, torch


@app.cell
def _(
    MODEL_NAME,
    UPSTREAM_NOTEBOOK_COMMIT,
    artifact_dir,
    gpu_name,
    json,
    metadata,
    mo,
    nvidia_smi,
    platform,
    torch,
):
    from huggingface_hub import model_info

    def pkg_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not installed"

    model_sha = model_info(MODEL_NAME).sha
    dependency_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "unsloth": pkg_version("unsloth"),
        "transformers": pkg_version("transformers"),
        "trl": pkg_version("trl"),
        "vllm": pkg_version("vllm"),
        "datasets": pkg_version("datasets"),
        "peft": pkg_version("peft"),
        "bitsandbytes": pkg_version("bitsandbytes"),
    }
    provenance = {
        "upstream_notebook_commit": UPSTREAM_NOTEBOOK_COMMIT,
        "model_name": MODEL_NAME,
        "model_revision": model_sha,
        "gpu_name": gpu_name,
        "cuda": torch.version.cuda,
        "dependency_versions": dependency_versions,
        "artifact_dir": str(artifact_dir),
    }
    (artifact_dir / "tysalim_run_metrics.json").write_text(
        json.dumps({"provenance": provenance, "nvidia_smi": nvidia_smi}, indent=2)
        + "\n"
    )
    mo.md(
        f"""
        ## Run provenance

        Filled from this Molab session. Re-run this cell after training if you
        want logs merged into `tysalim_run_metrics.json`.

        | Field | Value |
        |---|---|
        | Upstream notebook commit | `{UPSTREAM_NOTEBOOK_COMMIT}` |
        | Model | `{MODEL_NAME}` |
        | Model Hub SHA | `{model_sha}` |
        | GPU | `{gpu_name}` |
        | CUDA | `{torch.version.cuda}` |
        | Python | `{dependency_versions["python"]}` |
        | torch | `{dependency_versions["torch"]}` |
        | unsloth | `{dependency_versions["unsloth"]}` |
        | transformers | `{dependency_versions["transformers"]}` |
        | trl | `{dependency_versions["trl"]}` |
        | vllm | `{dependency_versions["vllm"]}` |
        | datasets | `{dependency_versions["datasets"]}` |
        """
    )
    return dependency_versions, model_sha, provenance


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Model, tokenizer, and chat template

        From the HF LLM course: a Transformer **architecture** is the layer
        graph; a **checkpoint** is a set of weights. Qwen2.5-3B-Instruct is a
        **decoder-only / autoregressive** checkpoint. Attention at position
        *t* may only see tokens `< t`. Pretraining is causal language
        modeling (predict the next token). Instruction tuning then teaches
        the model to answer as an assistant. That Instruct checkpoint is
        the starting policy here — Unsloth notes GRPO needs a model that
        can already emit a correct answer with non-zero probability.

        The tokenizer is not optional glue. Models consume integers, not
        strings (HF course ch.2). Qwen uses a **subword** vocabulary: common
        words stay whole, rarer spans split. Encoding is tokenize → ID list;
        decoding maps IDs back to text. Special chat tokens are part of that
        same vocabulary.

        `tokenizer.apply_chat_template(..., add_generation_prompt=True)`
        turns `{role, content}` dicts into the exact Qwen string the
        checkpoint was instruction-tuned on, then appends the assistant
        prefix so generation continues as the tutor/solver turn. vLLM
        samples only that continuation. Reward functions therefore read
        `completion[0]["content"]`, not the prompt.

        GRPO does not retrain the 4-bit base. LoRA (`r = 64`, seed 3407)
        adds low-rank adapters on the seven projection matrices that
        implement attention (`q,k,v,o_proj`) and the MLP
        (`gate,up,down_proj`). Those adapters **are** the trainable policy.
        """
    )
    return


@app.cell
def _(MODEL_NAME, torch):
    from unsloth import FastLanguageModel

    max_seq_length = 1024
    lora_rank = 64

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    loaded_revision = getattr(getattr(model, "config", None), "_name_or_path", MODEL_NAME)
    torch.cuda.empty_cache()
    return lora_rank, max_seq_length, model, tokenizer, loaded_revision


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## GSM8K prep and reward functions

        Copied from the pinned Unsloth notebook (Will Brown / @willccbb,
        also the GSM8K recipe in Unsloth's GRPO tutorial). GSM8K answers
        hide the numeric gold after `####`. The system prompt does **not**
        teach math; it only demands `<reasoning>` / `<answer>` tags. The
        tutorial is explicit: the dataset answer must not already contain
        the reasoning chain you want the policy to learn.

        Unsloth distinguishes a **verifier** (correct vs not) from a
        **reward function** (that check turned into a number). This notebook
        uses both as small programmatic rubrics — **RLVR**, not RLHF. There
        is no learned reward model and no human preference labels. TRL sums
        the five scores per completion:

        | Function | Signal |
        |---|---|
        | `xmlcount_reward_func` | 0.125 per well-placed XML tag; extra text after `</answer>` is fined |
        | `soft_format_reward_func` | 0.5 if `<reasoning>` then `<answer>` appear |
        | `strict_format_reward_func` | 0.5 if the whole completion matches the newline-exact template |
        | `int_reward_func` | 0.5 if the extracted answer is digits |
        | `correctness_reward_func` | 2.0 if that extract equals the gold string |

        Unsloth's 2+2 example is the same idea at smaller scale: digits beat
        letters, 4 beats 3. Format rewards exist so the model can get a
        non-zero signal before it ever matches GSM8K exactly — otherwise
        early groups are all-zero and GRPO has no relative ranking.
        """
    )
    return


@app.cell
def _():
    import re

    from datasets import Dataset, load_dataset

    SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

    XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def get_gsm8k_questions(split: str = "train") -> Dataset:
        data = load_dataset("openai/gsm8k", "main")[split]
        data = data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )
        return data

    dataset = get_gsm8k_questions()

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        q = prompts[0][-1]["content"]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print(
            "-" * 20,
            f"Question:\n{q}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def int_reward_func(completions, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return count

    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]

    return (
        SYSTEM_PROMPT,
        XML_COT_FORMAT,
        correctness_reward_func,
        dataset,
        extract_xml_answer,
        int_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        xmlcount_reward_func,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## GRPO: sampled completions → rewards → policy update

        Deep RL unit 1: an **agent** sees a **state**, takes an **action**,
        gets a **reward**, and aims to maximize expected return. Mapping
        that onto this cell:

        | RL term | Here |
        |---|---|
        | Agent / policy | Qwen2.5-3B-Instruct + LoRA (next-token distribution) |
        | Environment / state | One GSM8K chat prompt |
        | Action | One sampled assistant completion (up to 200 tokens) |
        | Reward | Sum of the five verifiers |
        | Return | That scalar; no multi-step discount inside the turn |

        PPO would also train a **value model** (baseline for advantage) and
        usually a **reward model**. DeepSeek's GRPO, as Unsloth summarizes
        it, drops both:

        1. For each prompt, vLLM draws a **group** of `num_generations=8`
           completions (the "Group" in GRPO).
        2. Each completion gets a scalar from the verifiers.
        3. Advantages are **group-relative**: Z-score the 8 rewards (subtract
           the group mean, divide by the group std). A completion only has
           to beat its siblings, not an absolute threshold. That replaces
           the value model.
        4. The policy gradient increases log-probability of tokens in
           high-advantage completions and decreases it for low-advantage
           ones, with a KL penalty toward the frozen Instruct reference so
           the policy cannot collapse. Optimizer here: `adamw_8bit`,
           `lr=5e-6`, cosine, 250 steps.

        Unsloth's "patience" point: if the chance of a useful completion is
        not literally zero, sampling will eventually produce one; GRPO then
        also **down-weights** the bad siblings so you are not waiting
        forever. The pinned notebook still warns that the logged `reward`
        column may sit near 0 for the first ~100 steps while format has not
        locked in. 250 steps is their published reference run, not a full
        GSM8K epoch.
        """
    )
    return


@app.cell
def _():
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=200,
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        output_dir="outputs",
    )
    return GRPOTrainer, training_args


@app.cell
def _(
    GRPOTrainer,
    correctness_reward_func,
    dataset,
    int_reward_func,
    model,
    soft_format_reward_func,
    strict_format_reward_func,
    tokenizer,
    training_args,
    xmlcount_reward_func,
):
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    train_result = trainer.train()
    return train_result, trainer


@app.cell
def _(artifact_dir, json, provenance, train_result, trainer):
    import matplotlib.pyplot as plt

    log_history = [
        row
        for row in trainer.state.log_history
        if "reward" in row or "loss" in row
    ]
    steps = [row.get("step", i) for i, row in enumerate(log_history) if "reward" in row]
    rewards = [row["reward"] for row in log_history if "reward" in row]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(steps, rewards, marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title("GRPO reward trace (250-step reference run)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    metrics_path = artifact_dir / "tysalim_run_metrics.json"
    _payload = {
        "provenance": provenance,
        "train_metrics": getattr(train_result, "metrics", {}),
        "log_history": trainer.state.log_history,
    }
    metrics_path.write_text(json.dumps(_payload, indent=2, default=str) + "\n")
    fig
    return fig, log_history, metrics_path, rewards, steps


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Inference: before vs after the LoRA

        Upstream generates **after** `trainer.train()`, first with
        `lora_request=None` (base Instruct policy) and then with the saved
        GRPO LoRA. The strawberry prompt is Unsloth's demo; the GSM8K row is
        a representative math item from the train split.
        """
    )
    return


@app.cell
def _(SYSTEM_PROMPT, dataset, model, tokenizer, train_result):
    from vllm import SamplingParams

    # Block this cell until the 250-step GRPO run finishes.
    _ = train_result

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )

    def generate(messages, lora_request):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text

    strawberry_user = [{"role": "user", "content": "How many r's are in strawberry?"}]
    strawberry_with_system = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "How many r's are in strawberry?"},
    ]
    gsm8k_example = dataset[0]
    gsm8k_prompt = gsm8k_example["prompt"]

    before_strawberry = generate(strawberry_user, lora_request=None)
    before_gsm8k = generate(gsm8k_prompt, lora_request=None)

    model.save_lora("grpo_saved_lora")
    lora_request = model.load_lora("grpo_saved_lora")

    after_strawberry = generate(strawberry_with_system, lora_request=lora_request)
    after_gsm8k = generate(gsm8k_prompt, lora_request=lora_request)

    generations = {
        "gsm8k_question": gsm8k_prompt[-1]["content"],
        "gsm8k_gold": gsm8k_example["answer"],
        "before_strawberry": before_strawberry,
        "after_strawberry": after_strawberry,
        "before_gsm8k": before_gsm8k,
        "after_gsm8k": after_gsm8k,
    }
    return generations, sampling_params


@app.cell
def _(artifact_dir, generations, json, metrics_path, mo):
    if metrics_path.exists():
        _payload = json.loads(metrics_path.read_text())
    else:
        _payload = {}
    _payload["generations"] = generations
    metrics_path.write_text(json.dumps(_payload, indent=2, default=str) + "\n")

    mo.md(
        f"""
        ### Strawberry (Unsloth demo)

        **Before LoRA** (`lora_request=None`, no system format prompt):

        ```
        {generations["before_strawberry"]}
        ```

        **After GRPO LoRA** (with the XML system prompt):

        ```
        {generations["after_strawberry"]}
        ```

        ### GSM8K train example 0

        **Question:** {generations["gsm8k_question"]}

        **Gold:** `{generations["gsm8k_gold"]}`

        **Before LoRA:**

        ```
        {generations["before_gsm8k"]}
        ```

        **After GRPO LoRA:**

        ```
        {generations["after_gsm8k"]}
        ```

        250 steps is a short reference run. Format often improves before
        exact-match accuracy; that is expected.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## How this differs from PedagogicalRL

        Sources: [Dinucu-Jianu et al., EMNLP 2025](https://arxiv.org/abs/2505.15607)
        and the official loop in
        [`src/classroom.py`](https://github.com/eth-lre/PedagogicalRL/blob/main/src/classroom.py).

        The paper's claim is that LLMs are optimized to **answer**, while
        tutoring requires **strategically withholding** the solution and
        scaffolding the student. They therefore run **online, on-policy,
        multi-turn GRPO** on the tutor's own dialogs — not SFT on a static
        teacher corpus and not single-turn preference data.

        Official classroom state machine:

        `START → TEACHER_TURN → STUDENT_TURN → … → JUDGE_TURN → GENERATE_SOLUTION → REWARD_TURN → END`

        Who opens the dialog is randomized (student attempt first, or tutor
        first). Optional `<think>` traces are hidden from the student. Only
        tutor tokens are trained; student turns are **masked** in the GRPO
        loss. Implementation defaults in `technical.md`: Qwen2.5-7B tutor,
        frozen Llama-3.1-8B student, Qwen2.5-14B-class judges, problems from
        filtered BigMath (numeric, medium-hard), not GSM8K.

        Reward is assigned **once, at the end of the conversation**
        (paper eqs. 2–4):

        - \(r_{sol}\): sample *K* final answers from the **student**,
          still conditioned on the dialog, and take the fraction that match
          gold. That is a post-dialog solve rate, not a tutor-free transfer
          test.
        - \(r_{ped}\): 1 only if **every** pedagogical judge accepts the
          transcript (helpfulness / no full-solution leakage).
        - Combined: \(r = r_{sol} + (r_{ped}-1)\lambda\). \(\lambda\) traces
          a Pareto frontier between student success and not leaking. A
          "hard" mode replaces the whole reward with \(-\lambda\) if any
          judge rejects.

        | | This notebook | PedagogicalRL |
        |---|---|---|
        | Job | **Solve** one GSM8K item in a single assistant turn | **Teach** through a multi-turn classroom dialog |
        | Policy | Qwen2.5-3B-Instruct + LoRA, Unsloth/TRL | Qwen2.5-7B tutor |
        | Other agents | None | Frozen student LLM + leakage/helpfulness judges |
        | Action | One completion of ≤200 tokens | Next tutor utterance in an MDP whose state is the dialog so far |
        | Reward | Format + integer + exact `####` match (pay for giving the answer) | \(r_{sol}\) minus a \(\lambda\) penalty when judges reject (pay for *not* giving the answer) |
        | Gold | Dataset string | No gold tutor script; student attempts + judges |
        | Data | `openai/gsm8k` train | Filtered BigMath |
        | GRPO group | 8 completions of the same prompt | Multiple full conversation rollouts per problem |

        Shared machinery: sample a group, score with verifiers/judges,
        update the policy relatively, no human tutor labels. Opposite
        objective: this run **rewards stating the GSM8K number**;
        PedagogicalRL **penalizes leaking it**. Do not ship this 3B adapter
        as a Socratic-tutor checkpoint. The paper also reports that
        post-dialog solve is dialog-conditioned, so it is not evidence the
        student could solve a fresh problem alone.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sources used for the writeups

        Assigned issue readings, not full courses:

        - PedagogicalRL paper, arXiv:2505.15607 (classroom MDP, \(r_{sol}\)/\(r_{ped}\)/\(\lambda\), online GRPO)
        - [eth-lre/PedagogicalRL](https://github.com/eth-lre/PedagogicalRL) README, `technical.md`, `src/classroom.py`, `train_rl.py`
        - [HF LLM course](https://huggingface.co/learn/llm-course/en/chapter1/1) ch.1 (decoder-only Transformers, checkpoints vs architecture) and [ch.2 tokenizers](https://huggingface.co/learn/llm-course/en/chapter2/4)
        - [HF Deep RL course](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction) unit 0 plus [the RL loop](https://huggingface.co/learn/deep-rl-course/en/unit1/rl-framework) (agent, state, action, reward)
        - [Unsloth RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) (GRPO vs PPO, group-relative advantages, RLVR)
        - [Unsloth GRPO tutorial](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo) (this GSM8K recipe)
        - [Molab GitHub-synced notebooks](https://docs.marimo.io/guides/molab/)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Do not commit weights

        `outputs/` (step-250 checkpoint) and `grpo_saved_lora/` are local Molab
        artifacts. `.gitignore` already excludes them. Commit this `.py` file
        and `tysalim_run_metrics.json` (provenance, logs, generation text only).
        """
    )
    return


if __name__ == "__main__":
    app.run()
