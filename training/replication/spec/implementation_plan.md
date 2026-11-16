# Implementation plan

The code lives in `training/` (GitHub). Run outputs go to the `socratictutor` Hugging Face org.

| Deliverable | File | Done when |
| --- | --- | --- |
| Paper prompts, verbatim | `tutor_rl/prompts.py` | Figures 9–14 copied; tests assert the judge prompt layout |
| Paper data split | `tutor_rl/data.py`, `configs/*.toml [data]` | Table 1 reproduced (`tab1_data_composition`) ✔ |
| Dialogue environment | `tutor_rl/episode.py` | student/tutor openers, 16 messages, K = 8, lockstep batching ✔ (tests) |
| Reward | `tutor_rl/rewards.py` | soft/hard λ, two judges × 2 samples, template terms ✔ (tests) |
| GRPO with μ = 2 | `tutor_rl/grpo.py` | clipped second pass, KL to base, linear LR, resume ✔ (CPU smoke) |
| Evaluation | `tutor_rl/evaluation.py` | Δ solve rate, Gemma leak %, paper reference table ✔ (tests) |
| Notebook | `tutor_rl_molab.py` | graphs, logs, eval charts; `marimo check` clean ✔ |
| Smoke run on molab | `configs/smoke.toml` | all three servers start; 2 steps train; eval writes results |
| Full run | `configs/molab.toml` | 150 steps; checkpoints on the Hub |
| Table 2 / Figure 1 | notebook section 6, then `replication/src/table2_from_eval.py` (to write) | `tab2_*` and `fig1_pareto` registered and compared |

**Order of targets:**

1. `tab1_data_composition` (now)
2. `fig2_difficulty` (CPU)
3. `tab2_base_tutor`, which needs only evaluation
4. `tab2_rl_lam075` and `fig1_pareto`, after the full run
