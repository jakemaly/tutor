# Tutor RL training (issue #46)

A resource-bounded trainer inspired by PedagogicalRL ([paper](https://aclanthology.org/2025.emnlp-main.15/), [authors' code](https://github.com/eth-lre/PedagogicalRL), read at `82178e3`). The tutor model is trained with multi-turn GRPO on dialogues with a simulated student. Its reward is the student's solve rate after the dialogue, or the minimum reward when a judge model says the tutor gave away the answer.

## Run it on molab

Open [`tutor_rl_molab.py`](tutor_rl_molab.py) in molab with the GPU turned on, paste a Hugging Face token that can read `meta-llama/Llama-3.1-8B-Instruct`, and work down the notebook: start the servers, check the sanity dialogue, train, evaluate, export. Run the `smoke` config once before `molab`. If the notebook runs outside a checkout, it downloads the branch named in its settings, so push the branch first.

The export zip contains everything later steps need: the LoRA adapter, config, data version, per-step metrics and rollouts, base-vs-trained eval, and provenance (repo commit, model, dataset and package versions, GPU).

## Layout

| Path | Purpose |
| --- | --- |
| `tutor_rl_molab.py` | marimo notebook: setup, servers, training with live plots, eval, export, Hub push |
| `configs/molab.toml` | Full run on one RTX PRO 6000 (96 GB) |
| `configs/smoke.toml` | The same models and code path in a few minutes |
| `prepare_data.py` | GSM8K at a pinned revision → `data/{train,eval}.jsonl` + `manifest.json` |
| `train.py` | CLI for the training loop (`--launch-servers`, `--resume`) |
| `evaluate.py` | CLI comparing policies on the same scenarios (`hf:base`, `hf:<adapter>`, or an endpoint) |
| `tutor_rl/episode.py` | Tutor ↔ student rollouts, batched in lockstep (provisional interface) |
| `tutor_rl/rewards.py` | Answer checking, leak judge, hard-λ reward, GRPO group advantages |
| `tutor_rl/policy.py` | Qwen3.5 tutor with LoRA as a batched `ChatModel`; records sampled tokens |
| `tutor_rl/grpo.py` | Multi-turn GRPO loss, training loop, checkpoints, resume |
| `tutor_rl/openai_chat.py` | Student/judge client for any OpenAI-compatible endpoint (stdlib only) |
| `tutor_rl/vllm_server.py` | Installs vLLM in its own venv and serves student and judge locally |
| `tutor_rl/evaluation.py`, `tutor_rl/export.py` | Base-vs-trained report; zip bundle, merged weights, Hub push |
| `tests/` | Fake-model tests (no GPU) |

## Compared with the paper

| | PedagogicalRL (`7b.yaml`) | `configs/molab.toml` |
| --- | --- | --- |
| Tutor | Qwen2.5-7B-Instruct, full fine-tune | Qwen3.5-9B (the app's model), thinking off, bf16 LoRA r=32 |
| Student | Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct, local vLLM |
| Judge | Qwen2.5-14B-Instruct-AWQ | Qwen2.5-14B-Instruct-AWQ, local vLLM |
| Batch | 16 problems × 8 samples | 6 problems × 4 episodes |
| Hardware | 4+ GPUs | One molab RTX PRO 6000, everything on one GPU |

Choices worth knowing about:

- **bf16 LoRA, not QLoRA.** Unsloth's Qwen3.5 guide advises against 4-bit training for this family because the quantization error is unusually large. The 9B model fits in bf16.
- **Plain transformers + PEFT, not Unsloth or TRL.** Qwen3.5 is a hybrid Gated DeltaNet model, and Unsloth's vLLM fast inference does not cover it. TRL's GRPOTrainer scores one completion per prompt. The loss in `grpo.py` is short and handles multi-turn episodes directly.
- **Exact tokens.** Each tutor turn becomes its own (prompt, completion) sample, with the prompt rendered by Qwen's chat template as the app's vLLM server renders it. The completion is the exact sampled token ids. Every turn in an episode shares that episode's advantage.
- **On-policy, one update per batch**, so the importance ratio is 1 and there is no clipping. KL (k3 estimator, β = 0.001) is measured against the base model by disabling the adapter, so no second copy of the model is needed.
- **vLLM in its own venv** (`vllm==0.32.0`, as in `deployment/modal_qwen.py`), so its pins never conflict with the training stack.

## Run the tests

```sh
uv run --project training pytest
```

## Still to decide

- Whether the "Sprint 1 episode/model interfaces" named in the issue exist somewhere. If they do, `tutor_rl/episode.py` should adopt them.
- Week 2: serve `final/adapter` (or the merged checkpoint from the notebook's Hub push) through the vLLM path, then run `evaluate.py` against the endpoints.
