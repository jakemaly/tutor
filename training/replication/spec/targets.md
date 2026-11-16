# Targets

Paper: Dinucu-Jianu, Macina, Daheim, Hakimi, Gurevych, Sachan. *From Problem-Solving to Teaching
Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning.* EMNLP 2025
(aclanthology.org/2025.emnlp-main.15, arXiv 2505.15607, CC BY 4.0).

**Scope set by the user (2026-11-16):**

- The tutor stays Qwen3.5-9B. The paper's Qwen2.5-7B is not used.
- Problems, prompts, and training step size follow the paper.
- Training runs on molab (one RTX PRO 6000).

So the paper's absolute numbers are a reference, not a pass bar. The replicated claim is the paper's central one: RL with the paper's reward moves a tutor from high leakage and a small Δ solve rate toward low leakage and a larger Δ solve rate.

| target | what | acceptance | status |
| --- | --- | --- | --- |
| `tab1_data_composition` | Table 1: per-source counts and Llama-3.1-8B solve rate | numeric: counts exact, rates ≤ 0.05 pp | MATCHED (counts exact, rates ≤ 0.005 pp) |
| `tab2_base_tutor` | Table 2, untrained tutor row: Δ solve rate (pp), leaked solutions (%) with the Gemma-3-27B judge | qualitative: base leaks more than the trained tutor; paper Qwen2.5-7B base 11.3 / 29.3 | ACTIVE (needs a molab eval of the base tutor) |
| `tab2_rl_lam075` | Table 2, RL λ = 0.75 row | qualitative: Δ solve up and leak down vs. base; paper 25.3 / 10.6 | PENDING (molab train + eval) |
| `fig1_pareto` | Figure 1: tutors in the (leak %, Δ solve) plane | qualitative: trained point up and to the left of base | PENDING (made by `plots.eval_tradeoff`) |
| `fig2_difficulty` | Figure 2: solve-rate bucket distribution | distributional: bucket shares within 1 pp | PENDING (CPU-only; script still to write) |
| `fig3_lambda_sweep` | Figure 3: metrics vs. λ (7 runs) | — | SKIPPED: 7 full runs are beyond the compute budget; the config supports any λ |
| `tab2_hard_and_think` | RL-hard and +think rows | — | SKIPPED: extra runs; `hard = true` is supported, thinking tags are not wired in |
| `tab2_baselines` | SFT, MDPO, SocraticLM, LearnLM, GPT-4o, other open models | — | SKIPPED: other systems, outside issue #46 |
| `tab3_reasoning` | MMLU / GSM8K / MATH500 retention | — | SKIPPED: not requested |
| `tab4_mathtutorbench` | MathTutorBench and Ped-RM | — | SKIPPED: needs the benchmark's reward model; not requested |

Ambiguities that affect the in-scope targets are in `assumptions_and_unknowns.md` (H1–H12).
