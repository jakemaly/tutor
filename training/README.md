# Tutor RL training (issue #46)

Trains the app's tutor model, Qwen3.5-9B, to guide students without giving answers away. It follows PedagogicalRL ([paper](https://aclanthology.org/2025.emnlp-main.15/), CC BY 4.0): multi-turn GRPO on dialogues with a simulated student (Llama-3.1-8B-Instruct), on the paper's Big-Math problems, with the paper's prompts.

**Reward:** a judge (Qwen2.5-14B-Instruct-AWQ) checks every dialogue for answer leaks and for unhelpful teaching. A rejected dialogue gets −1; otherwise the reward is the share of the student's solutions afterwards that are correct, plus 0.1 if the tutor ended the dialogue.

**Result to look for:** compared with the untrained tutor on the same held-out problems, the trained tutor leaks the answer less, and the student solves as often or more.

## Run it on molab

Setup (Hugging Face org and token, molab GPU, what goes where) is in [WORKFLOW.md](WORKFLOW.md). In short:

1. Open [`tutor_rl_molab.py`](tutor_rl_molab.py) in molab with the GPU on and paste a Hugging Face token.
2. Start the servers and check the sanity dialogue.
3. Train, then evaluate untrained vs. trained.
4. Week 2, in a fresh session: run sections 1 and 2, skip 3 to 5, and click section 6. It serves the base model and the trained adapter with vLLM on the molab GPU (the same OpenAI-compatible server the app uses) and compares them on the same scenarios. To serve it on Modal for the app instead, see [docs/modal-serving.md](../docs/modal-serving.md#serve-the-trained-tutor-issue-46).

Run the `smoke` config once before `molab`. Checkpoints, metrics, chat logs and results upload to a private repo in the `socratictutor` org as they are saved, and a new session resumes from the Hub.

## Layout

| Path | Purpose |
| --- | --- |
| `tutor_rl_molab.py` | marimo notebook: setup, servers, training with a live plot and chat logs, untrained-vs-trained evaluation, served base-vs-trained evaluation, export |
| `configs/molab.toml` | The run: 60 steps × 24 dialogues on one RTX PRO 6000 (96 GB) |
| `configs/smoke.toml` | The same models and code path in a few minutes |
| `prepare_data.py` | The Big-Math split at a pinned revision → `data/{train,eval}.jsonl` + `manifest.json` |
| `train.py` | CLI for the training loop (`--launch-servers`, `--resume`) |
| `evaluate.py` | CLI comparing policies on the same scenarios (`hf:base`, `hf:<adapter>`, or an endpoint) |
| `tutor_rl/prompts.py` | Tutor, student and judge prompts (from the paper) |
| `tutor_rl/episode.py` | Tutor ↔ student dialogues, batched in lockstep; student or tutor opens |
| `tutor_rl/rewards.py` | Answer checking, the judges, the reward, GRPO group advantages |
| `tutor_rl/policy.py` | Qwen3.5 tutor with LoRA as a batched `ChatModel`; records sampled tokens |
| `tutor_rl/grpo.py` | Multi-turn GRPO loss, training loop, `train.log`, checkpoints, resume |
| `tutor_rl/evaluation.py` | Untrained vs. trained: answer-leak rate and student solve rate |
| `tutor_rl/vllm_server.py` | Installs vLLM in its own venv and serves the student, the judge and (week 2) the tutor locally |
| `tutor_rl/gpu_budget.py` | Keeps everything on the GPU under 80% of its memory |
| `tutor_rl/hub.py`, `tutor_rl/export.py` | Hub uploads and resume; provenance and zip |
| `tutor_rl/openai_chat.py` | Endpoint clients: the student and judge (chat), and the served tutor (`ServedTutor`, same prompt as training) |
| `tutor_rl/config.py`, `data.py` | Config loading, problem files |
| `tests/` | Fake-model tests (no GPU) |

## Design notes

- **Tutor:** bf16 LoRA r=32 on Qwen3.5-9B. Unsloth advises against QLoRA for this family.
- **Plain transformers + PEFT, not Unsloth or TRL.** Qwen3.5 is a hybrid Gated DeltaNet model that Unsloth's fast inference doesn't cover, and TRL's GRPO trainer is single-turn. Each tutor turn becomes its own sample with the exact sampled tokens, and every turn in a dialogue shares that dialogue's advantage.
- **KL** (β = 0.001) is measured against the base model by switching the adapter off, so no second copy of the model is needed.
- **GPU memory:** the student and judge vLLM servers take 15% each, and the tutor is hard-capped at 50%, so 20% stays free. Generation batches are sized by tokens, so long dialogues don't run out of memory.
- **vLLM** runs in its own venv (`vllm==0.32.0`) and is stopped as a whole process group, so no GPU memory is left behind.

## Run the tests

```sh
uv run --project training pytest
```
