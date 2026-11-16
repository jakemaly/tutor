# TODO

## Current phase
implementation complete (CPU-verified); GPU targets pending

## Active target
tab2_base_tutor

(Evaluate the untrained Qwen3.5-9B tutor on the 500 test problems: notebook section 6, Gemma leak judge.)

## Acceptance gates
- [x] Inventory the paper source (`inspect-paper`)
- [x] Enumerate every target in `spec/reproduction_matrix.csv` with an acceptance mode
- [x] Replace placeholder sections in `spec/*.md`
- [x] Keep exactly one ACTIVE target
- [x] tab1_data_composition: run-wrapper artifact, provenance, comparison, report embed
- [ ] Smoke run on molab (`configs/smoke.toml`): servers, 2 steps, eval
- [ ] tab2_base_tutor
- [ ] Full run (`configs/molab.toml`, 150 steps) -> tab2_rl_lam075, fig1_pareto
- [ ] fig2_difficulty (CPU; read bucket edges off the rendered figure first)
- [ ] `validate-completion`

## Open unknowns
See `spec/assumptions_and_unknowns.md` (H1 resolved; H2-H14 open, with checks listed there).

## Completed
- Bootstrap, paper intake, spec.
- Trainer, prompts, data, reward, evaluation, notebook graphs and logs (tests + CPU smoke).
- Table 1 reproduced exactly.
