# Math audit

**Setting.** A dialogue is a sequence of utterances u_1..u_T. The state is the history s_t = u_<t, and the action a_t = u_t is the next utterance. Tutor actions are sampled from π_θ; student actions come from a fixed LLM, so the transitions are stochastic. The tutor ends the dialogue with `<end_of_conversation>`, or it stops at 16 messages.

**Reward** (Eqs. 1–3 and appendix B), given once per dialogue:

- r_sol = (1/K) Σ_k 1[ŝ_k = s], with K = 8 post-dialog student solutions. Implemented in `rewards.solve_rate` and `rewards.is_correct`.
- r_ped = Π_m 1[J_m accepts]. The judges are the leakage and helpfulness prompts, each sampled twice, so 4 verdicts must all be OK. Implemented in `rewards.run_judges` and `score_episodes`.
- Soft: r = r_sol + (r_ped − 1)·λ + r_templ. Hard: r = −λ if r_ped = 0. Implemented in `rewards.combine`.
- r_templ = 0.1·1[ended by tutor] − 0.5·1[any tutor turn hit max tokens]. The think-tag terms are 0 because thinking is off. Implemented in `rewards.template_reward`.

**Advantage.** For the G = 8 rollouts of one problem, A_i = (r_i − mean(r)) / (std(r) + 1e-4), using the unbiased std as TRL does (`rewards.group_advantages`). Every token of every tutor turn in dialogue i gets A_i: no discount, and student and system tokens are masked.

**Objective** (`grpo.GRPOTrainer._update`). The sum runs over the tutor tokens y of all samples in the batch, and N is the total number of tutor tokens:

    L = (1/N) Σ_y [ −min(ρ_y A, clip(ρ_y, 1−ε, 1+ε) A) + β·k3_y ]
    ρ_y = π_θ(y) / π_old(y)
    k3_y = exp(log π_ref(y) − log π_θ(y)) − (log π_ref(y) − log π_θ(y)) − 1

- μ = 2 optimizer steps per batch.
- On pass 1, π_old = π_θ exactly, so ρ = 1 and the gradient is the plain policy gradient. log π_old is recorded from that same forward pass (detached), which saves a separate forward pass.
- π_ref is the base model with the LoRA adapter disabled, computed once per sample per batch.
- Log-probs use the sampling temperature T = 1.0, so they match the distribution the tokens were sampled from (`completion_logprobs`).

**Checked numerically (CPU, tiny random Qwen3.5 with the real tokenizer):**

- With ρ = 1 on pass 1 and equal-length turns, the policy-gradient loss is 0 because group advantages sum to 0.
- At a large LR, the second pass clips 35–93% of tokens and KL turns positive.
- Scheduler and adapter state survive a save and resume exactly (4 updates after 2 steps).
