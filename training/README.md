# Tutor RL training (issue #46)

A resource-bounded trainer inspired by PedagogicalRL ([paper](https://aclanthology.org/2025.emnlp-main.15/), [authors' code](https://github.com/eth-lre/PedagogicalRL), read at `82178e3`). The tutor model is trained with GRPO on multi-turn dialogues with a simulated student. Its reward is the student's solve rate after the dialogue, minus a penalty when a judge model says the tutor gave away the answer.

## Layout

| Path | Status | Purpose |
| --- | --- | --- |
| `configs/small.toml` | draft | One-GPU run: 3B QLoRA tutor, student and judge over an OpenAI-compatible endpoint |
| `tutor_rl/episode.py` | working, provisional interface | Tutor ↔ student rollout and the student's post-dialogue test |
| `tutor_rl/rewards.py` | working | Answer checking, leak judge, hard-λ reward, GRPO group advantages |
| `tutor_rl/prompts.py` | working | Tutor, student, and judge prompts (written for this repo) |
| `prepare_data.py` | TODO | GSM8K → versioned `train/eval.jsonl` and a manifest |
| `train.py` | TODO | QLoRA policy, rollouts, multi-turn GRPO update, checkpoint saving |
| `evaluate.py` | TODO | Base vs. trained on the same fixed scenarios |
| `tests/` | working | Fake-model tests for the episode loop and rewards |

## Scaled down from the paper

| | PedagogicalRL (`7b.yaml`) | This config |
| --- | --- | --- |
| Tutor | Qwen2.5-7B-Instruct, full fine-tune | Qwen2.5-3B-Instruct, 4-bit QLoRA |
| Student | Llama-3.1-8B-Instruct (FP8) | Shared endpoint (for example the Modal vLLM deployment) |
| Judge | Qwen2.5-14B-Instruct-AWQ | Same endpoint as the student |
| Batch | 16 problems × 8 samples | 4 problems × 4 episodes |
| Hardware | 4+ GPUs | One Molab RTX 6000 or Modal GPU |

## Run the tests

```sh
uv run --project training pytest
```

## Still to decide

- Whether the "Sprint 1 episode/model interfaces" named in the issue exist somewhere. If they do, `tutor_rl/episode.py` should adopt them.
- Which models play the student and judge, and the compute budget.
- Where checkpoints live (Hugging Face Hub or a Modal volume). Weights are never committed.
