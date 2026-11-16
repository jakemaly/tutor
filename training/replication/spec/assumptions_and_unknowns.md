# Assumptions and unknowns

Each line is a detail the paper does not pin down (or one we change on purpose), the value we use, and why. "Paper" means stated in the LaTeX source (arXiv 2505.15607). The authors' code was not read for this replication (`author_code_policy = forbid_by_default`).

## Deliberate deviations (user scope)

- **D1 Tutor model.** Paper: Qwen2.5-7B-Instruct. Here: Qwen3.5-9B @ `c2022362`, thinking off (the app's model).
- **D2 LoRA instead of a full fine-tune.**
  - Paper: full fine-tune at lr 5e-7 with `paged_adamw_8bit`.
  - Here: bf16 LoRA r=32, alpha 32, on attention, DeltaNet and MLP projections, at lr 5e-6 with torch AdamW.
  - Why: a full fine-tune does not fit next to the vLLM student and judge on one 96 GB GPU. LoRA's best learning rate is usually about 10× full fine-tuning's.
  - Optimizer state for LoRA is small, so the 8-bit optimizer buys nothing.
- **D3 Test-time leak judge.** Paper: Gemma3-27B (precision not stated). Here: `RedHatAI/gemma-3-27b-it-quantized.w4a16` @ `2b537554`, 4-bit so it fits beside the others.
- **D4 Hardware.** Paper: 4× A100, ~48 h. Here: one RTX PRO 6000, spread over several 12-hour molab sessions with Hub resume.

## Hypotheses (paper silent)

- **H1 Dataset split.**
  - The paper describes the filter but not the split procedure.
  - `rd211/Big-Math-RL-Verified-Filtered` @ `80714c39` (published by the first author, Oct 2025) has 10,000 train and 500 test problems. Its per-source counts and mean solve rates equal Table 1 (see `tab1_data_composition`).
  - **Resolved:** it is the paper's split.
- **H2 Post-dialog prompt layout.**
  - Figure 10 puts `{{ conversation }}` before the instruction text.
  - Here the student's own chat history (system prompt with the problem, then the turns) is followed by one user message with Figure 10's text.
  - Alternative: render the transcript into a single user message. Untested.
- **H3 Student opening turn.**
  - In the student-initiated scenario the student gets only its system prompt (Figure 11), with no extra "show your attempt" message.
- **H4 Tutor-initiated prompt on Qwen3.5.**
  - Qwen3.5's chat template raises "No user query found" for system-only prompts.
  - `policy.prompt_ids` renders system plus generation prompt, which is the layout Qwen2.5's template produces.
- **H5 Answer checking.**
  - The paper doesn't say how correctness is judged. Here: the last `\boxed{}` in a solution, checked symbolically against the gold answer with math-verify 0.9.0. A solution with no boxed answer is wrong, since Figure 10's prompt requires one.
  - Table 1's solve rates come from the dataset annotation, so this rule only affects the reward and the evaluation.
- **H6 Judge sampling temperature.** Paper: "sample twice from each" judge. Here: 0.7 (it must be above 0 for two samples to differ). A dialogue is accepted only if all 4 samples say OK, matching the product over judges in Eq. 3.
- **H7 Student temperature.** Not stated. Here: 0.6 (Llama-3.1's default generation config), for chat turns and for the K = 8 solutions.
- **H8 Token limits.**
  - Not stated. Tutor 256 tokens per turn; the 0.5 truncation penalty applies beyond it.
  - Student 256 tokens per chat turn, 1024 for pre- and post-dialog solutions.
  - Judges 512.
- **H9 Learning-rate schedule and clipping.**
  - Not stated. The paper adapts TRL's GRPOTrainer, so we use TRL's defaults: linear decay, no warmup, clip ε = 0.2, Adam (0.9, 0.999), no weight decay, max grad norm 1.0.
- **H10 Loss normalization.**
  - Here: sum over all tutor tokens in the batch, divided by the token count (token-level mean, every tutor turn weighted equally per token).
  - The paper says only that tutor utterances are treated equally, with no discount.
- **H11 Number of steps.**
  - Paper: "~300 policy updates", "roughly 20% of the training data".
  - Here: 150 batches × μ 2 = 300 updates. That covers 2,400 problems (24%).
- **H12 Leak metric at test time.** Here: one sample from the eval judge at temperature 0, with Figure 13's prompt. A dialogue counts as leaked if that sample says REJECT.
- **H13 Judge prompt examples.** Figure 13 elides two of its four examples ("[1 more ...]"). Only the two printed examples are used, renumbered 1 and 2.
- **H14 Pre-dialog solve rate in Table 2.**
  - Here it is measured with Figure 9's prompt and K = 8 samples per problem.
  - The dataset's 64-sample annotation is saved next to it in `eval/pre_dialog.jsonl` for comparison.

## Open checks to run on molab

- The smoke run's seconds per step gives the real wall-clock estimate for the full run.
- H2/H3: read a few sanity dialogues to confirm the student behaves like a student, attempting the problem when it opens.
- H6: in `rollouts/`, check how often the two samples of a judge disagree.
