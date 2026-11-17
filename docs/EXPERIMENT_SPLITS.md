# Experiment Splits

This repository uses fixed experiment buckets so development, historical production, and submission evidence cannot silently drift into one another. Historical/development buckets are physically separated under `90_ARCHIVE/`; active submission evidence remains in `generated_submission/`.

## Submission-authoritative evidence set

The current manuscript is locked to:

- manifest: `configs/experiment_manifest_submission_primary.csv`
- run ledger: `generated_submission/master_runs.csv`
- artifact root: `generated_submission/`

The submission-primary matrix contains **10,800 completed runs**:

- 36 benchmark instances
- 3 operating scenarios
- 20 paired seeds
- 5 primary methods: `EDE`, `StdDE`, `ALNS_MS`, `HGS_MS`, `ILS_MS`

Each primary method contributes 2,160 runs. A1/A2/A3 are not part of submission-primary inference.

### Temporal/regime semantics of this matrix

For the public Solomon/Homberger inputs, the production loader preserves the native benchmark horizon: all three shift/regime labels use the same `[0,H]` availability interval. This matrix is therefore an **overlapping full-horizon operating-regime special case**, not a sequential three-shift experiment. Scenario labels alter emission multipliers across vehicle/regime/zone dimensions; they do not impose different fleet shares. Vehicle type is route-level and persistent physical vehicle identity across regime labels is outside the submission-primary evidence.

## Submission benchmark composition

### Small tier — Solomon, 100 customers

`C101`, `C104`, `C109`, `C201`, `R101`, `R104`, `R109`, `R201`, `RC101`, `RC104`, `RC108`, `RC201`

### Medium tier — Homberger–Gehring, 200 customers

`C1_2_1`, `C1_2_5`, `C2_2_1`, `C2_2_5`, `R1_2_1`, `R1_2_5`, `R2_2_1`, `R2_2_5`, `RC1_2_1`, `RC1_2_5`, `RC2_2_1`, `RC2_2_5`

### Large tier — Homberger–Gehring, 400 customers

`C1_4_1`, `C1_4_5`, `C2_4_1`, `C2_4_5`, `R1_4_1`, `R1_4_5`, `R2_4_1`, `R2_4_5`, `RC1_4_1`, `RC1_4_5`, `RC2_4_1`, `RC2_4_5`

The three customer-structure classes (`clustered`, `mixed`, `random`) are evenly represented within each size tier. Li & Lim support remains available in the codebase, but no Li & Lim instance is part of the authoritative 10,800-run submission ledger and the manuscript must not imply otherwise.

## Historical production ledger

`90_ARCHIVE/02_GENERATED_HISTORY/generated_real/master_runs.csv` contains the completed **17,280-run historical production ledger**. It includes the five primary methods plus 6,480 historical `A1_NoSeed`, `A2_NoJDE`, and `A3_NoLNS` rows.

Those ablation rows predate the 2026-09-02 single-factor remediation. They are retained for provenance only and are excluded from component-effect claims and submission-primary inferential statistics. No new 6,480-run campaign is required for the current manuscript scope.

## Fixed development and preflight buckets

### Dev tuning split

- outputs: `90_ARCHIVE/02_GENERATED_HISTORY/generated/dev_broad_route_traj/`
- role: development-direction and acceptance-quality diagnostics only

### Medium comparator split

- manifest: `90_ARCHIVE/11_CONFIG_AND_DEV_HISTORY/experiment_manifest_medium_compare.csv`
- outputs: `90_ARCHIVE/02_GENERATED_HISTORY/generated/master_runs_medium_compare.csv`, `90_ARCHIVE/02_GENERATED_HISTORY/generated/medium_compare/`
- role: comparator sanity checking during development only

### Paper preflight subset

- manifests: `90_ARCHIVE/11_CONFIG_AND_DEV_HISTORY/paper_main3_real_batches/`
- reports: `90_ARCHIVE/02_GENERATED_HISTORY/generated/paper_main3_real_preflight_report.txt`
- role: verify selected paper-facing implementations and benchmark loading before production

### Submission preflight subset

- reports: `90_ARCHIVE/02_GENERATED_HISTORY/generated/submission_full_real_preflight_report.txt`
- role: historical full-manifest benchmark readiness check; not the current inferential evidence source

## Tuning outputs

Focused development/tuning artifacts (including the archived `90_ARCHIVE/02_GENERATED_HISTORY/generated/paper_main3_real/` tree and its sweep directories) are diagnostic only. They must not be pooled into `generated_submission/`.

## Promotion rule

Only artifacts regenerated from `generated_submission/master_runs.csv` may support final manuscript percentages, significance values, and submission tables. Historical or tuning outputs may be cited only for provenance or implementation history, never as interchangeable evidence.

The manuscript and all derived documentation must preserve the same scope statement: the completed public-benchmark evidence validates the full-horizon regime special case only. Sequential non-overlapping shift claims require a different experiment and are not inferred from the 10,800-run ledger.

## Secondary operational-semantics suite

`configs/operational_shift_stress_locked.csv` is a separate 18-run synthetic stress suite with three contiguous non-overlapping shifts. It is excluded from the 10,800-run primary comparator design and from Friedman/Wilcoxon inference. Its role is limited to validating implemented shift/duty semantics. `configs/solomon_equal_3shift_feasibility_audit.csv` records why naive equal partitioning of native Solomon instances is not used as a substitute primary benchmark.


## Secondary local parameter-robustness suite

`configs/parameter_sensitivity_locked.csv` contains 45 secondary runs: C101/R101/RC101 × three paired seeds × five OAT configurations under S3, with a 3-s wall-clock cap and 300 nominal evaluation guard. It is excluded from the 10,800-run primary comparator design and from its 108-block inference. The suite tests only local robustness of LNS cadence and repair budget around the frozen E-DE default; normalization weights, overtime penalties, other parameters, and full-budget/tier-wide sensitivity remain outside this secondary screen.


## Secondary short-budget anytime suite

`configs/anytime_profile_locked.csv` contains 48 secondary runs: C101/R101/RC101 × E-DE/StdDE × paired seeds 1/2 × wall-clock caps 1/2/4/8 s under S3, with a nominal outer evaluation guard of 1000. It is excluded from the 10,800-run primary comparator design and from its 108-block inference. The suite provides a descriptive short-budget anytime profile only: median score by instance/cap, paired wins/ties/losses, and log2-time normalized score AUC. It does not establish a general convergence rate, asymptotic behavior, medium/large-tier behavior, or full-budget superiority.
