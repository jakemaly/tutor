# Tutor RL training (issue #46)

This is a replication of PedagogicalRL: Dinucu-Jianu et al., *From Problem-Solving to Teaching Problem-Solving*, EMNLP 2025 ([paper](https://aclanthology.org/2025.emnlp-main.15/), CC BY 4.0). The trained tutor is the app's model, Qwen3.5-9B. It is trained with multi-turn GRPO on dialogues with a simulated student, on the paper's problems, with its prompts, reward, and batch size. The reward is

    r = r_sol + (r_ped − 1)·λ + r_templ

- `r_sol` is the student's solve rate over K = 8 solutions written after the dialogue.
- `r_ped` is 1 only if both judges (answer leakage and helpfulness, each sampled twice) accept the dialogue.
- `r_templ` is +0.1 if the tutor ends the dialogue, and −0.5 if any tutor turn was cut off.

The replication state lives in [`replication/`](replication/):

- `spec/targets.md`: what we try to match.
- `spec/reproduction_matrix.csv`: the status of each target.
- `spec/assumptions_and_unknowns.md`: every value the paper doesn't state, and what we chose.

## Run it on molab

Setup (Hugging Face org and token, molab GPU, the AI assistant, what goes where) is in [WORKFLOW.md](WORKFLOW.md). In short: open [`tutor_rl_molab.py`](tutor_rl_molab.py) in molab with the GPU on and paste a Hugging Face token. Then work down the notebook: start the servers, check the sanity dialogues, train, watch the graphs and logs, evaluate, and export. Run the `smoke` config once before `molab`.

Every checkpoint, plus the metrics, rollouts, logs (`logs/train.log` and the vLLM logs), and eval results, uploads to a private repo in the `socratictutor` org as soon as it is saved. The problem set goes to a dataset repo. A new molab session resumes from the newest checkpoint on the Hub.

## Layout

| Path | Purpose |
| --- | --- |
| `tutor_rl_molab.py` | marimo notebook: setup, servers, training with live graphs, graphs and logs viewer, eval with the paper's comparison chart, export, Hub push |
| `configs/molab.toml` | The replication run (paper values marked `# paper`) |
| `configs/smoke.toml` | The same models, prompts and code path at a size that runs in minutes |
| `prepare_data.py` | The paper's Big-Math split at a pinned revision → `data/{train,eval}.jsonl` + `manifest.json` |
| `train.py` | CLI for the training loop (`--launch-servers`, `--resume`) |
| `evaluate.py` | CLI for the paper's Table 2 metrics, comparing policies on the same scenarios (`hf:base`, `hf:<adapter>`, or an endpoint) |
| `tutor_rl/prompts.py` | The paper's six prompts, verbatim (appendix, Figures 9–14) |
| `tutor_rl/episode.py` | Tutor ↔ student rollouts, batched in lockstep, with student- or tutor-initiated openings |
| `tutor_rl/rewards.py` | Answer checking (math-verify), the two judges, the soft or hard λ reward, the template reward, group advantages |
| `tutor_rl/policy.py` | Qwen3.5 tutor with LoRA as a batched `ChatModel`; records sampled tokens and truncation |
| `tutor_rl/grpo.py` | Multi-turn GRPO with μ clipped passes, training loop, `train.log`, checkpoints, resume |
| `tutor_rl/plots.py` | Training dashboard and eval charts used by the notebook |
| `tutor_rl/openai_chat.py` | Student/judge client for any OpenAI-compatible endpoint (stdlib only) |
| `tutor_rl/vllm_server.py` | Installs vLLM in its own venv and serves the student, the judge, and the eval judge locally |
| `tutor_rl/hub.py` | Background checkpoint uploads, resume from the Hub, dataset push |
| `tutor_rl/evaluation.py`, `tutor_rl/export.py` | Δ solve rate and leak report against the paper's numbers; provenance, zip, merged weights |
| `replication/` | Paper-replication harness: spec, targets, assumptions, report (the paper source is gitignored) |
| `WORKFLOW.md` | molab + Hugging Face setup and how to run an experiment |
| `tests/` | Fake-model tests (no GPU) |

## Compared with the paper

| | Paper | `configs/molab.toml` |
| --- | --- | --- |
| Tutor | Qwen2.5-7B-Instruct, full fine-tune, lr 5e-7, paged AdamW 8-bit | **Qwen3.5-9B**, thinking off, bf16 LoRA r=32, lr 5e-6, AdamW |
| Student | Llama-3.1-8B-Instruct, FP8, vLLM | same |
| Judges | Qwen2.5-14B-Instruct AWQ; leakage + helpfulness prompts, 2 samples each | same |
| Test-time leak judge | Gemma3-27B | Gemma-3-27B-it, 4-bit (w4a16) |
| Data | Big-Math, Llama solve rate 1–60%, 10,000 train / 500 test | the same split (`rd211/Big-Math-RL-Verified-Filtered`) |
| Dialogue | ≤ 16 messages; student or tutor opens (uniform); K = 8 | same |
| Batch | 16 problems × 8 rollouts, μ = 2, β = 0.001, T = 1.0 | same |
| Length | ~300 policy updates (~20% of the train set) | 150 steps × μ 2 = 300 updates (24%) |
| Reward | r_sol + (r_ped − 1)·λ + r_templ, several λ | same, λ = 0.75 (the paper's balanced model); `hard = true` for the hard variant |
| Hardware | 4× A100 80 GB, ~48 h | one RTX PRO 6000 96 GB, several 12-hour molab sessions |

**The target:** the paper's in-domain Table 2. With λ = 0.75 the RL tutor reaches **25.3** Δ solve rate and **10.6%** leaked solutions, against 11.3 and 29.3% for its untrained base. The tutor model differs here, so the absolute numbers will too. The claim we can test is the shape: after training, Δ solve rate rises and leaks fall relative to the untuned Qwen3.5-9B, landing near the paper's λ = 0.75 point.

Choices worth knowing about:

- **LoRA instead of a full fine-tune.** A full fine-tune of a 9B model doesn't fit next to the student and judge on one GPU. LoRA usually wants about 10× the full fine-tune learning rate, hence 5e-6. Unsloth's Qwen3.5 guide advises against QLoRA for this family.
- **Plain transformers + PEFT, not Unsloth or TRL.** Qwen3.5 is a hybrid Gated DeltaNet model, and Unsloth's vLLM fast inference does not cover it. TRL's GRPOTrainer scores one completion per prompt. The loss in `grpo.py` handles multi-turn episodes directly.
- **Exact tokens.** Each tutor turn becomes its own (prompt, completion) sample, with the prompt rendered by Qwen's chat template as the app's vLLM server renders it. The completion is the exact sampled token ids. Every turn in an episode shares that episode's advantage, with no discounting, as in the paper.
- **μ = 2 without an extra forward pass.** The first pass over a batch is on-policy (ratio 1). It records the old and reference log-probs, and the second pass uses the PPO clipped ratio against them (ε = 0.2). KL (k3 estimator, β = 0.001) is measured against the base model by disabling the adapter.
- **When the tutor opens.** Qwen3.5's chat template rejects a prompt with no user message, which is the tutor's first turn when it opens. `policy.prompt_ids` then renders the system prompt plus the generation prompt alone, as the paper's Qwen2.5 template does.
- **vLLM in its own venv** (`vllm==0.32.0`, as in `deployment/modal_qwen.py`), so its pins never conflict with the training stack.

## Run the tests

```sh
uv run --project training pytest
```

## Still to decide

- Whether the "Sprint 1 episode/model interfaces" named in the issue exist somewhere. If they do, `tutor_rl/episode.py` should adopt them.
- Week 2: serve `final/adapter` (or the merged checkpoint from the notebook's Hub push) through the vLLM path, then run `evaluate.py` against the endpoints.
