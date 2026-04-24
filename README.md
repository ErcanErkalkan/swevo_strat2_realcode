# SWEVO Strategy-2 Real-Code Experiment Package

This package contains a **real, runnable reference implementation** for the manuscript's claimed
MS-VRPTW comparator set:

- EDE
- StdDE
- ALNS_MS
- HGS_MS
- ILS_MS
- A1_NoSeed
- A2_NoJDE
- A3_NoLNS

The code is written to match the manuscript logic as closely as possible from the available text:

- feasibility-first decoding,
- bounded overtime at return-to-depot,
- deterministic feasible seeding,
- random-key DE / jDE,
- boundary-focused LNS,
- ALNS destroy-repair search,
- HGS-style giant-tour population search with split decoding,
- ILS with boundary perturbations,
- matched-budget run ledger,
- validator / aggregation / statistics / LaTeX table generation.

## Important scope note

This is a **reference implementation**, not a claim that the paper's final numerical results have already
been reproduced. The original uploaded ZIP did not contain a complete live optimizer codebase or benchmark
instance files, so this package provides:

1. real optimizer code,
2. deterministic benchmark/scenario generation from the manifest,
3. the full experiment pipeline,
4. reproducible run logging,
5. manuscript-facing table/macro generation.

That means the package now removes the earlier “contract stub” problem. It does **not** magically certify
that the manuscript's reported percentages are already correct; those must still come from actual runs.

## Main study scale

The main manifest defines **17,280 planned runs**:

36 instances × 3 scenarios × 8 methods × 20 paired seeds.

A smaller developer manifest is also included:

- `configs/experiment_manifest_micro.csv`
- Split roles and promotion rules are documented in `docs/EXPERIMENT_SPLITS.md`.

## Package layout

## Real benchmark files

If public benchmark files are available, place them under:

- `data/benchmarks/solomon/`
- `data/benchmarks/homberger/`
- `data/benchmarks/li_lim/`

The loader now auto-detects Solomon-like text instances (`CUST NO.`, `XCOORD.`, `READY TIME`, `DUE DATE`, `SERVICE TIME`).
If no file is found for a manifest row, the suite falls back to deterministic synthetic generation.
Preflight checks now distinguish a locally missing file from a manifest row that names a non-public benchmark id such as `RC109` in the Solomon-100 family.
If that happens, `scripts/propose_benchmark_repairs.py` writes proposal inventory/manifest files without mutating the canonical configs.


- `configs/` full manifest, method registry, budgets, metrics schema, stats plan
- `src/swevo_suite/benchmark.py` deterministic benchmark/scenario builder
- `src/swevo_suite/solver.py` objective evaluation, decoder, repair, local search, archive, search kernels
- `src/swevo_suite/comparators/` live comparator adapters
- `scripts/` matrix generation, run execution, validation, aggregation, stats, LaTeX output
- `generated/` run outputs and manuscript-facing artifacts

## Real-run quick start

### 1. Micro validation run

```bash
cd swevo_strat2_realcode
PYTHONPATH=src python scripts/run_manifest.py run \
  --manifest configs/experiment_manifest_micro.csv \
  --output generated/master_runs_micro.csv \
  --budget-override 12 \
  --overwrite --progress
python scripts/validate_master_runs.py generated/master_runs_micro.csv
```

### 2. Build summaries and manuscript tables

```bash
python scripts/finalize_submission_state.py
```

Equivalent step-by-step commands:

```bash
python scripts/check_benchmark_inventory.py
python scripts/propose_benchmark_repairs.py
python scripts/aggregate_results.py
python scripts/run_stats.py
python scripts/build_latex_tables.py
python scripts/build_claim_macros.py
python scripts/build_claim_evidence_map.py
python scripts/write_pipeline_audit.py
python scripts/check_submission_gates.py
```

### 3. Full experiment run

```bash
python scripts/prepare_paper_run.py \
  --methods EDE StdDE ALNS_MS HGS_MS ILS_MS A1_NoSeed A2_NoJDE A3_NoLNS \
  --prefix submission_full_real
PYTHONPATH=src python scripts/run_manifest.py run \
  --manifest configs/experiment_manifest_full.csv \
  --output generated/master_runs.csv
```

## Algorithm mapping

### EDE

- random-key encoding (2-column keys)
- deterministic feasible seeding
- jDE self-adaptation on `F` and `CR`
- feasibility-first selection
- bounded repair
- boundary-focused LNS
- mini-restart on diversity collapse

### StdDE

- same random-key / decode / repair layer
- no deterministic seed
- no jDE
- no LNS

### ALNS_MS

- destroy/repair schedule search
- operators: random, worst, related, shift-border removal
- feasibility-first reinsertion and local search

### HGS_MS

- giant-tour population
- order crossover
- split decoding through the same MS-VRPTW decoder
- survivor selection with simple diversity pressure

### ILS_MS

- single-solution search
- local search to local optimum
- shift-border perturbations / swap perturbations
- probabilistic acceptance

### Ablations

- `A1_NoSeed`: EDE without deterministic seeding
- `A2_NoJDE`: EDE with fixed `F/CR`
- `A3_NoLNS`: EDE without boundary-focused LNS

## Submission-critical outputs

Before manuscript submission, these must be regenerated from real runs:

- `generated/master_runs.csv`
- `generated/summary_by_method.csv`
- `generated/summary_by_method_accepted_only.csv`
- `generated/master_stats.csv`
- `generated/master_stats_accepted_only.csv`
- `generated/table_feasibility_summary.tex`
- `generated/table_feasibility_summary_accepted_only.tex`
- `generated/table_friedman_omnibus.tex`
- `generated/table_friedman_omnibus_accepted_only.tex`
- `generated/table_posthoc_holm.tex`
- `generated/table_posthoc_holm_accepted_only.tex`
- `generated/claim_macros.tex`
- `generated/claim_evidence_map.csv`

## Non-negotiable rules

- Accepted final rows must have zero final violations.
- Final performance tables must not mix accepted outputs with violation counts.
- Search diagnostics and final performance remain separate layers.
- If ALNS/HGS/ILS appear in the paper, they must remain in the manifest and stats outputs.
- Manuscript percentages must come from generated artifacts, not hand-typed values.
