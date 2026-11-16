# ML workflow: molab, Hugging Face, GitHub

## What goes where

| GitHub (`jakemaly/tutor`) | Hugging Face (`socratictutor` org) |
| --- | --- |
| Notebooks and training `.py` code | Run outputs: metrics, rollouts, eval results |
| Configs (`training/configs/*.toml`) | Checkpoints (LoRA adapters + optimizer state) |
| Experiment instructions (this file, README) | Logs (`train.log`, vLLM server logs) |
| Small manifests | Datasets (one folder per data version) |
| | Models (final adapters, merged weights) |

Never commit weights, run outputs, or datasets. `training/outputs/` is in `.gitignore`.

## One-time setup

### Hugging Face

1. Accept the org invite the team lead shared, and check that you appear under *Members* at <https://huggingface.co/socratictutor>.
2. Open <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and accept the licence. Approval is usually quick but not instant. The student model can't download until it is granted.
3. Create a token under *Settings → Access Tokens*:
   - Simplest: token type **Write**.
   - Tighter: **Fine-grained**, with *Read access to contents of all public gated repos you can access* plus *Write access to contents/settings of all repos* under the `socratictutor` org.
   - Treat the token like a password. Paste it into the notebook's token field; never put it in a cell's code, a config, or a commit.

### molab

1. Open the notebook straight from GitHub:
   `https://molab.marimo.io/github/jakemaly/tutor/blob/<branch>/training/tutor_rl_molab.py`
   This makes a notebook synced to GitHub: pushes to that branch show up in molab. Alternatively, paste the GitHub URL into the new-notebook dropdown on the molab home page.
2. Turn on the GPU with the notebook specs button in the header (RTX PRO 6000, 96 GB).
3. Limits to plan around:
   - GPU sessions last at most **12 hours**.
   - Notebooks idle for **90 minutes** are shut down.
   - Files over **1 GB** are not kept after shutdown.
   - The notebook therefore uploads every checkpoint to the Hub as it is saved, and resumes from the Hub.
4. *Remote storage* (Files panel in the left sidebar → *Remote Storage* → *Add remote storage*): once you've pasted your token and the settings cell has run, `HF_TOKEN` is in the kernel and molab offers Hugging Face as a quick add. Use it to browse run repos and datasets without leaving the notebook. The marimo docs also list GitHub through fsspec (`GithubFileSystem`) but don't say whether it works in molab. For code, the GitHub-synced notebook above is the dependable route.

### The molab AI assistant

- **Built-in chat:** molab includes free AI assistance on fast open-source models, with no key needed. Open the chat panel in the sidebar and choose its *Notebook access* level:
  - *Read only*: it can explain cells and errors. Start here.
  - *Suggest edits*: it proposes changes for you to accept.
  - *Full access*: it can edit and run code. Avoid this while a long training run is going.
- **Your own model:** marimo's settings (*Settings → AI*) take providers such as Anthropic, OpenAI, or any OpenAI-compatible endpoint, with an API key. The molab docs don't say whether molab exposes this, so check the settings menu. Costs go on your key.
- **A coding agent on your own machine:** the notebook menu has *Pair with an agent*. It works with Claude Code, Codex, and OpenCode after installing marimo's pair skill locally, and gives you a prompt to paste into the agent.
- **Prompts that work well here:** be specific and name the file or cell. For example:
  - "Explain what `completion_logprobs` in tutor_rl/grpo.py computes, line by line."
  - "Step 14's reward dropped to −1 for every episode. Read rollouts/step-00014.jsonl and tell me why."

  Ask it to explain before it edits.

## Running an experiment

1. Change things through a config, not the code: copy `configs/molab.toml` to a new name, edit it, commit it, and push.
2. In the notebook, pick the config and set *Hub run repo name* to a new name for each experiment. Reusing a name continues (or overwrites) that run. Name runs after what changed, for example:
   - `pedagogicalrl-qwen35-9b-lam075` (the default)
   - `pedagogicalrl-qwen35-9b-hard-lam1`, the paper's hard variant: `[reward] hard = true`, `lam = 1.0`
   - `pedagogicalrl-qwen35-9b-lam0`, which gives away answers freely, as in the paper's λ = 0 row
3. Run `smoke` first after any code change. Its *seconds per step* times the step ratio estimates the full run. The full run has 64× the dialogues per step and 4× the turns, so expect several 12-hour sessions.
4. Watch section 5 (*Training graphs and logs*) during the run: reward and r_sol should rise, leakage rejects should fall, and KL should stay small.
5. After the run, evaluate (section 6). Put the Hub link, the eval table, and the trade-off chart in the issue or PR, and commit any config or instruction changes to GitHub.

## Where a run's files end up

- Run repo `socratictutor/<run name>` (private):
  - `checkpoints/step-NNNNN/` (adapter, optimizer state, config, data manifest)
  - `final/adapter`
  - `metrics.jsonl`, `rollouts/`, `logs/`, `eval/`, `provenance.json`
- Dataset repo `socratictutor/pedagogicalrl-bigmath` (name set by `[data] hub_dataset`): `<data_version>/train.jsonl`, `eval.jsonl`, `manifest.json`
- Merged model (optional): `socratictutor/<run name>-merged`
