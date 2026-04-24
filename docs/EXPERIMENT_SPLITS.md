# Experiment Splits

This repo uses fixed experiment buckets so tuning, comparison, and submission preparation do not silently drift.

## Current canonical benchmark set

- `configs/benchmark_inventory.csv`
- `configs/experiment_manifest_full.csv`

The canonical mixed Solomon small set is:

- `RC101`
- `RC104`
- `RC108`
- `RC201`

`RC109` is not part of the public Solomon-100 benchmark family and has been removed from the canonical inventory/manifest.

## Root generated package

The checked-in root manuscript-facing package currently uses the strongest fully populated all-comparator real-run sample set:

- source runs: `generated/master_runs_sample.csv`
- promoted root runs: `generated/master_runs.csv`

This root package is intentionally separate from the blocked/noisy tuning outputs so LaTeX tables, claim macros, stats, and the claim-evidence map all refer to one consistent artifact set.

## Fixed split roles

### Dev tuning split

- outputs: `generated/dev_broad_route_traj/`
- purpose: fast EDE direction finding and acceptance-quality checks

### Medium comparator split

- manifest: `configs/experiment_manifest_medium_compare.csv`
- outputs: `generated/master_runs_medium_compare.csv`, `generated/medium_compare/`
- purpose: comparator sanity checks on a broader real-run subset

### Paper preflight subset

- manifests: `configs/paper_main3_real_batches/`
- reports: `generated/paper_main3_real_preflight_report.txt`
- purpose: verify the paper-facing `EDE / ILS_MS / StdDE` slice is runnable with real benchmarks

### Submission preflight subset

- reports: `generated/submission_full_real_preflight_report.txt`
- purpose: verify the full comparator manifest is benchmark-ready before long production runs

## Tuning outputs

Focused C101 wall-60 EDE tuning outputs live under:

- `generated/paper_main3_real/`
- `generated/paper_main3_real/sweeps/`

These are diagnostic artifacts only. They should not be mixed into the root manuscript-facing package unless explicitly promoted.
