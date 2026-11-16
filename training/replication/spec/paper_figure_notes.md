# Paper figure notes

Written from the LaTeX source and captions. The figure PDFs in the source tree are reference only.

- **Figure 1 (`fig1_pareto`):** a scatter of Δ solve rate (y, %) against leaked solutions (x, %) on the Big-Math test set. It shows the RL models across λ, which trace a Pareto front, plus the baselines. The reproduction (`tutor_rl/plots.eval_tradeoff`) plots the same axes: the paper's Table 2 points in grey, and our base and trained Qwen3.5-9B in colour, with an arrow from base to trained.
- **Figure 2 (`fig2_difficulty`):** a histogram of problems by Llama-3.1-8B solve-rate bucket, mostly in the 1–10% bucket. It can be computed from the dataset's `llama8b_solve_rate`. Bucket edges must be read off the rendered figure before comparing.
- **Figure 3 (`fig3_lambda_sweep`):** three panels against λ: Δ solve rate, leak rate, and Ped-RM (micro). Skipped.
- **Figures 4–8:** example conversations. Qualitative and not targets. The notebook's rollout viewer shows our equivalents.
