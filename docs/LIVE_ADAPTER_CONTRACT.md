# Live Adapter Contract

The adapter layer is now implemented with live reference solvers for:

- EDE
- StdDE
- ALNS_MS
- HGS_MS
- ILS_MS
- A1_NoSeed
- A2_NoJDE
- A3_NoLNS

Each comparator exposes:

`run(plan: RunPlan) -> RunResult`

## Required semantic guarantees

- `accepted_final == 1` only if all final hard violations are zero,
- `strict_duty_final == 1` only if accepted and overtime is zero,
- `compute_wh > 0`,
- final rows use the canonical `RunResult` schema,
- run-level notes describe solver mode and unusual events.

## Shared fairness layer

All solvers must use the same canonical semantics for:

- customer service uniqueness,
- capacity feasibility,
- hard customer time windows,
- bounded overtime at return-to-depot,
- accepted vs. infeasible final outputs,
- canonical score logging.

## Method-specific logs

Custom method-level logs are allowed, but manuscript-facing outputs must always be normalized into
`master_runs.csv` using the canonical schema.
