# Can a student use the method on a second problem?

For [Issue #47](https://github.com/jakemaly/tutor/issues/47), we compare a baseline tutor with the same tutor prompted to elicit a next step, a justification, and a reusable method from the student. Both are assessed on the original problem and an unseen variant. The intervention is a change to tutoring at inference time. The masked transfer test is a measurement applied equally to both tutors. The experiment requires no weight updates.

## Two choices for the learning signal

| Design decision | Eduardo | Sherpa |
| --- | --- | --- |
| What earns credit? | Improvement on a held-back variant after hiding the tutor's turns. Positive gain is normalized by remaining headroom and discounted for token use. A factual error or solution handover incurs a fixed penalty. | Improvement on the original problem after accepted exchanges. Only turns that pass both gates receive improvement credit. Advantages use a turn-level leave-one-out baseline. |
| What does the student see? | Frozen Llama-3.1-8B-Instruct makes its own errors. At test time the student retains its own utterances, with placeholders for the tutor. | Frozen Qwen3-1.7B receives instructions that pass both an answer-disclosure check and a preference check. On rejection the teacher gets scripted feedback and the student's context stays fixed. |
| How was it evaluated? | Held-out math dialogues, component ablations, MathTutorBench and TutorMoments. The ablation distinguishes the contribution of transfer to generalization from that of the gates to reducing handover. | 528 held-out MATH problems across four seen and three unseen archetypes, MathTutorBench, and paired response preferences from 96 high-school teachers. Teachers rated responses; student achievement in a classroom was not measured. |
| Compute | DPPO across 4B–27B tutors. A typical 100-step run uses eight GH200s for 10–12 hours plus separate student and judge servers. Each trajectory adds 16 post-test samples and six judge calls. | Rank-16 LoRA on an 8B teacher, 1,500 updates, 16 problems × eight trajectories per update, up to ten tutoring turns, online gates and eight samples per pre/post condition. Up to three private preparation attempts per problem. |
| What fits this repo? | The existing dialogue object already identifies speakers and the solver accepts separate prompts. Four paired targets are enough to exercise the mechanism. | A faithful test changes the simulator, routing of rejected messages and turn-level credit. A persona prompt alone would skip the mechanism that makes the preferences consequential. |

Sources: Macina, Kapur and Sachan, *The Assistance Dilemma*, §§3–5 and Appendices A–F ([full paper](https://arxiv.org/html/2610.06446v1)). Xu, Zhang, Wang, Chen and Yang, *Sherpa*, §§2–3 and Appendices A–D ([full paper](https://arxiv.org/html/2610.08778v1)). Workload units differ across papers, so the table does not infer a speed or dollar-cost ratio.

## Repository fit and controls

The inspected `dev` commit was `5740f14`. Its training and evaluation scripts are placeholders. The overlapping [PR #53](https://github.com/jakemaly/tutor/pull/53) supplies the Python interfaces used here, pinned at `2a19bd9d4aea66a01647784331d94f2ac495dcc9`. Test with that checkout until its interfaces land on `dev`. PR #31 is an earlier onboarding notebook. The other open PRs inspected (#54, #57, #58 and #63) concern skills, inference, or UI.

In #53, `grpo.GRPOTrainer.step` scores rollouts and normalizes rewards within each problem group. The smoke configuration uses a hard rejection penalty, source solve rate and a termination/truncation term. `run_episodes` alternates tutor and student views and collects full-context source solutions. The leakage and helpfulness judges receive the transcript. The existing evaluator pairs problems, openers, and sampling seeds across policies. Its post-test uses the original problem and retains the tutor's wording.

The companion runner reuses the smoke config, problem reader, opener selection, episode runner, answer checker and judges. It runs scenarios serially for timing. Each pair shares a source, opener, checkpoint, sampling budget, seed and one cold test on the target. Only the experimental tutor's system instruction changes. Both arms get the same student-only post-test and source leakage judge; the tutor never sees the target or its key. Source IDs and hashes guard against accidental dataset changes.

The primary measure is masked target accuracy minus cold target accuracy, in percentage points. With two samples per problem the estimated ceiling is unstable, so we report the raw difference. We serialize speaker-labelled notes into one user message and exclude the source system prompt. This differs from Eduardo's alternating masked chat, normalized reward, factual gate and efficiency decay. The existing helpfulness judge evaluates interaction style. It cannot establish factual correctness. The score is available for evaluation; the prototype does not feed it into GRPO.

## Four held-out variants

The source fixture is #53's first four `test` rows of [Big-Math-RL-Verified-Filtered](https://huggingface.co/datasets/rd211/Big-Math-RL-Verified-Filtered/tree/80714c393abc52de4025c3e7fc6acd30f376be52). The following checks establish each variant's key. Their difficulty relative to the originals has not been calibrated.

| Source row | Transfer key | Verification |
| --- | ---: | --- |
| 0: constrained minimum | 8/3 | Set s = a+b+c. The objective is 2s/9 + 8/s, with s ≥ 6. AM–GM gives 8/3, attained at a=b+c=3. |
| 1: dice bound | 128 | The maximum of u²+v² is 8²+8² and that roll has positive probability. |
| 2: maximum on positive reals | −6 | t+25/t ≥ 10, with equality at t=5. |
| 3: exponential equation | 8 | Exponent zero gives y=7,−9. Base one gives y=2,5. Base minus one gives candidates 3,4, of which only 3 has even exponent. The five valid roots sum to 8. |

## Reproduce

From this branch, use Python 3.11+ and an isolated environment. The dependency worktree supplies the existing trainer code without adding its commits to this PR.

```sh
git fetch origin refs/pull/53/head
git worktree add --detach ../tutor-53 2a19bd9d4aea66a01647784331d94f2ac495dcc9
export PYTHONPATH="$(cd ../tutor-53/training && pwd)"
cfg="$PYTHONPATH/configs/smoke.toml"
python -m pip install pytest datasets
python "$PYTHONPATH/prepare_data.py" --config "$cfg" --run-dir /tmp/tutor47
python -m pytest -q evals/near_transfer
```

For the CPU pilot, obtain the [Qwen2.5-0.5B-Instruct Q4_K_M GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/main/qwen2.5-0.5b-instruct-q4_k_m.gguf). SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`. The pilot uses the upstream plain/boxed numeric checker without optional math-verify. This can miss mathematically equivalent answers expressed in other forms.

```sh
python -m pip install llama-cpp-python==0.3.16
python evals/near_transfer/compare.py --config "$cfg" \
  --data /tmp/tutor47/data/eval.jsonl --gguf ./qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --output /tmp/tutor47/cpu-results.json
```

For the team's models, install the tokenizer dependency from #53 (`transformers==5.23.0`) and `math-verify==0.9.0`. Set the base URL and model name for each of `TUTOR_BASE`, `TRAIN_STUDENT` and `TRAIN_JUDGE` using the project's `<PREFIX>_BASE_URL` and `<PREFIX>_MODEL` variables, plus optional API keys. Point both arms at the same tutor checkpoint using `TUTOR_BASE`. Omit `--gguf` from the command above and choose a fresh output filename. The served tutor expects #53's completions interface and tokenizer. `--seed` overrides the smoke seed of 1234. Output includes all attempts, transcripts, raw judge decisions, per-arm samples and elapsed time. Shared cold-test cost is recorded once per scenario and repeated on both paired rows for reference.

## CPU pilot results

Both arms used Qwen2.5-0.5B-Instruct Q4_K_M in every role on four CPU threads. Tutor and student turns were capped at 96 tokens, final solutions and judge replies at 192, with four dialogue messages and two final attempts. The upstream smoke opener sequence was tutor, student, student, student. This pilot used the team's scenarios, prompts and scorer with a substitute model. No trained project checkpoint was evaluated.

| Primary measurement | Baseline | Elicitation |
| --- | ---: | ---: |
| Cold transfer solves (shared) | 0/8 | 0/8 |
| Original solves with full dialogue | 0/8 | 0/8 |
| Masked transfer solves | 0/8 | 0/8 |
| Masked gain over cold | 0 pp | 0 pp |
| Dialogues flagged by upstream leakage scorer | 0/4 | 1/4 |
| Explicit REJECT votes for leakage | 0/8 | 0/8 |
| Completions excluding shared cold probes | 48 | 48 |
| Wall time excluding shared cold probes | 195.8 s | 205.7 s |

The eight shared cold completions took 28.6 s. No GPU time or paid model calls were used. Wall time includes generation and grading, excludes model loading, and reflects one serial run with baseline first. The 9.9 s difference does not establish a latency effect.

All 24 cold and masked transfer attempts lacked a complete boxed answer. One experimental dice response correctly reasoned that R must be at least 128, then stopped mid-sentence. The exact scorer gave it zero. The experimental flag on the exponential equation came from a leakage judge response lacking a parseable decision; #53 treats that as rejection. There was no explicit REJECT vote in either arm. Some accepted tutor turns contained mathematical errors. These observations limit what the primary measurements establish about solving ability and answer handover. The [raw report](../../evals/near_transfer/results/cpu-pilot.json) contains the eight episodes and judge outputs.

### Answer-format sensitivity check

After seeing the primary scoring floor, we replayed the same student notes with a request for only the final boxed answer. The tutor, transcripts, targets, seeds, number of draws and 192-token solver limit were fixed. This additional analysis changes only the probe instruction. It should be interpreted separately from the primary comparison.

```sh
python evals/near_transfer/retest.py --gguf ./qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --report evals/near_transfer/results/cpu-pilot.json --output /tmp/tutor47/short-results.json
```

Both arms again scored 0/8 masked solves, with a shared cold score of 0/8. The shorter instruction elicited complete boxes in 4/8 baseline and 2/8 experimental attempts, all incorrect. For example, each arm produced 48 for the eight-sided dice problem whose answer is 128. A format change recovered some scorable responses without establishing a transfer advantage. This replay used 24 additional student completions. [Replay outputs](../../evals/near_transfer/results/short-probe.json)

## Limits and validation

The test measures immediate transfer through student-authored context. A student can echo an earlier hint, and the notes can carry an obsolete source constraint into the new problem. A frozen model has no persistent learning state. Four items with two samples each cannot support a general ranking of tutor policies. The shared small model also judges its own teaching. The project's Qwen tutor, Llama student and separate Qwen judge require configured endpoints; the hosted run stopped at the missing `TUTOR_BASE_BASE_URL` and `TUTOR_BASE_MODEL`. The student and judge endpoint variables were also absent.

The six focused tests cover masking, policy isolation, paired seeds/openers, sample counts, scoring and rejection, source verification and short-answer prompts. Repository lint, TypeScript type checking, production build and all 37 backend tests passed; this environment needed `socksio` installed locally for the existing HTTP client test. Ruff and mypy with `--check-untyped-defs` passed on the experimental implementation. The detached #53 training suite on Python 3.12 had 23 passes and one failure in checkpoint-upload staging. Its unchanged `test_checkpoint_upload_stages_checkpoint_and_run_files` passed when rerun on Python 3.13. Docker Compose checks were unavailable because the Docker executable was absent.
