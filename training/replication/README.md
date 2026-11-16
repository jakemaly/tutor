# From Problem-Solving to Teaching Problem-Solving (PedagogicalRL, EMNLP 2025)

This case study was scaffolded for a LaTeX-first paper replication workflow.

## Core commands

```bash
python3 scripts/paper_harness.py status --project-dir .
python3 scripts/paper_harness.py inspect-paper --project-dir .
python3 scripts/paper_harness.py validate-completion --project-dir .
bash scripts/reproduce_all.sh
```

## Expected workflow

1. Inspect the paper sources and inventory the TeX tree.
2. Fill out the spec files under `spec/`.
3. Keep `spec/reproduction_matrix.csv` and `todo.md` in sync, including per-target `acceptance_mode`.
4. Mark exactly one ACTIVE target at a time.
5. Use `track-run`, `register-target-artifact`, and comparison evidence before moving any target to MATCHED.
6. Register baseline targets only after a real paper-method implementation exists under project code, config, and spec trace files.
7. Use exact visual matching only for targets that truly require it; convergence curves and other stochastic summaries should usually use numeric or structural acceptance.
8. Move a target to MATCHED only after the artifact exists, provenance is recorded, and `report/main.tex` embeds it.
9. Treat the paper as done only when `validate-completion` passes.

## Case study id

- Paper slug: `pedagogical-rl`
