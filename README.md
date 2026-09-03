# SWEVO Strategy-2 Real-Code Experiment Package

This package contains a real, runnable reference implementation for the sustainability-aware shift-indexed VRPTW study. The mathematical formulation permits shift-specific availability intervals, while the submission-primary public-benchmark experiment uses a benchmark-preserving special case in which the three shift/regime labels all span the native horizon `[0,H]`.

## Submission-primary scope

The final submission-primary evidence set uses **five methods**:

- EDE
- StdDE
- ALNS_MS
- HGS_MS
- ILS_MS

The authoritative primary ledger contains **10,800 completed real-benchmark runs**:

36 instances × 3 scenarios × 5 methods × 20 paired seeds.

The completed historical 17,280-run ledger additionally contains A1_NoSeed, A2_NoJDE, and A3_NoLNS. Those 6,480 historical ablation rows were generated before the single-factor definitions were locked on 2026-09-02. They are retained for provenance, but they are **not used for submission-primary inferential statistics or component-effect claims**.

A new 6,480-run ablation campaign is therefore **not required for the claims made in the current manuscript**. The paper evaluates the integrated EDE system and explicitly does not claim how much each individual EDE component contributes.

## Final submission archive and author declarations

The exact owner-confirmed submission-primary reproducibility snapshot is distributed with the manuscript as Supplementary Material:

- `SWEVO_FINAL_REPRODUCIBILITY_SUPPLEMENT_2026-09-03.zip`
- SHA-256: `3aad983271e0e1cb6e7b4c62e38949906861ccf156b6e74722b6d500b6576a95`

The public submission-scope branch is `submission-2026-09-02`. The release-style frozen branch ref is `release/v0.2.0-submission-rc1`; it is a branch ref, not a Git tag.

Author declarations confirmed on 2026-09-03:
- Funding: no specific grant.
- Competing interests: none.
- CRediT: Conceptualization; Methodology; Software; Validation; Formal analysis; Investigation; Data curation; Writing - original draft; Writing - review & editing; Visualization.
- Acknowledgements: none.
- Originality/concurrent submission: the manuscript is not under consideration elsewhere.

## Primary temporal/regime semantics

The 10,800-run public-benchmark experiment **does not partition the Solomon/Homberger planning horizon into three consecutive driver shifts**. The loader preserves the native benchmark feasibility region by creating three shift/regime labels whose availability intervals are all `[0,H]`, where `H` is the original benchmark horizon. The labels therefore represent overlapping operating regimes rather than empirically validated sequential handoffs.

All three vehicle types are available under every regime label. Scenarios change vehicle-specific emission multipliers together with regime and zone multipliers; they do **not** change fleet-share proportions. Vehicle type is selected at route level and no persistent physical vehicle identity is tracked across regime labels. Consequently, the current evidence supports the integrated method on a shift-indexed/full-horizon-regime special case, not cross-shift vehicle reuse or sequential driver-shift claims.

## Production hardware environment

The production campaign was executed on a **Dell Vostro 3888** desktop with an **Intel Core i5-10400** CPU (6 physical cores / 12 threads, 2.90 GHz base frequency), **16 GB RAM** (2×8 GB at 2400 MHz), integrated Intel UHD Graphics 630, and **Windows 11 Enterprise 64-bit build 26200**. Storage comprised a 256 GB SK hynix NVMe SSD and a 480 GB Toshiba TR200 SATA SSD. BIOS: Dell 2.35.1; motherboard: Dell 0RM5DR. The optimizer is CPU-based and did not use the integrated GPU for search computation.

The seed-level ledger does not embed a per-run hardware fingerprint. Runtime comparisons are therefore interpreted within this documented workstation and tier-matched protocol rather than as cross-platform guarantees. `compute_wh` remains a runtime × assumed-tier-power model proxy and is **not** hardware-metered electrical energy.

## Implemented algorithmic elements

- feasibility-first decoding
- bounded overtime at return-to-depot
- deterministic feasible seeding
- random-key DE / jDE
- boundary-focused LNS and route intensification
- ALNS destroy-repair search
- HGS-style giant-tour population search with split decoding
- ILS with boundary perturbations
- paired-seed, tier-matched wall-clock production protocol
- validator / aggregation / statistics / LaTeX table generation

## Primary evidence files

The submission package uses:

- `configs/experiment_manifest_submission_primary.csv`
- `generated_submission/master_runs.csv`
- `generated_submission/summary_by_method.csv`
- `generated_submission/summary_by_method_accepted_only.csv`
- `generated_submission/master_stats.csv`
- `generated_submission/master_stats_accepted_only.csv`
- `generated_submission/table_feasibility_summary.tex`
- `generated_submission/table_friedman_omnibus.tex`
- `generated_submission/table_posthoc_holm.tex`
- `generated_submission/claim_macros.tex`
- `generated_submission/claim_evidence_map.csv`
- `generated_submission/robustness_by_scenario.csv`
- `generated_submission/robustness_by_structure.csv`
- `generated_submission/tier_native_endpoint_reductions.csv`
- `generated_submission/secondary_endpoint_stats.csv`
- `generated_submission/SUBMISSION_EVIDENCE_SCOPE.md`

The full historical ledger remains available separately and must not be substituted for the submission-primary ledger when regenerating manuscript statistics.

For a frozen submission/release branch, historical generated directories such as `generated/`, `generated_real/`, `generated_pilot/`, and `generated_smoke/` should not be left beside `generated_submission/` in a way that could be mistaken for primary evidence. Preserve them in a historical archive/tag or external evidence archive, while keeping `generated_submission/` as the only manuscript-facing generated namespace.

## Real benchmark files

Public benchmark files are stored under:

- `data/benchmarks/solomon/`
- `data/benchmarks/homberger/`
- `data/benchmarks/li_lim/`

The loader auto-detects Solomon-like text instances. If a configured benchmark file is not available locally, the inventory/preflight tools identify the condition rather than silently changing the submission-primary evidence set.

The authoritative 10,800-run submission ledger itself uses **12 Solomon 100-customer instances and 24 Homberger--Gehring instances (12 at 200 customers and 12 at 400 customers)**. Li & Lim support remains in the codebase but is not part of the current submission-primary evidence and must not be described as such in the manuscript.

## Algorithm mapping

### EDE

- random-key encoding
- deterministic feasible seeding
- jDE self-adaptation of `F` and `CR`
- EDE donor/current-to-best strategy
- feasibility-first selection
- bounded repair
- trajectory/deep intensification
- boundary-focused LNS
- route-ALNS endgame
- diversity restart

### StdDE

- common objective/feasibility interface
- random initialization
- fixed `F/CR`
- rand/1-style DE donor
- no EDE-specific LNS/intensification stack

### ALNS_MS

- destroy/repair schedule search
- random, worst, related, and duty-horizon-boundary removal
- feasibility-first reinsertion and local search

### HGS_MS

- giant-tour population
- order crossover
- split decoding through the common shift-indexed VRPTW evaluation interface
- survivor selection with diversity pressure

### ILS_MS

- single-solution search
- feasibility-first local improvement
- duty-horizon-boundary/swap perturbations
- probabilistic acceptance

### Code-level component variants

The code also exposes controlled single-concept variants derived from the current EDE configuration:

- `A1_NoSeed`: remove informed deterministic seeding only
- `A2_NoJDE`: fix `F/CR` while retaining the EDE donor strategy and all other non-jDE components
- `A3_NoLNS`: remove the LNS family while retaining non-LNS EDE components

These definitions are kept for reproducibility and future component studies. They are not part of the current submission-primary statistical comparison.

## Measurement and statistical semantics

- The production comparison is **wall-clock matched by tier**. `eval_budget` is a nominal outer search-step guard and is not presented as a complete count of every nested objective evaluation.
- `compute_wh` is a **model-based compute-energy proxy** calculated as runtime × assumed tier power / 3600. It is not hardware-metered electrical energy.
- The assumed tier powers are 52 W (small), 86 W (medium), and 128 W (large).
- Pairwise Wilcoxon inference first collapses the 20 paired seeds to the median within each `instance_id × scenario_id` block.
- Friedman and Wilcoxon-Holm therefore use the same **108 independent blocks**.
- Search diagnostics and final performance are reported separately.
- Production hardware is documented above; the ledger does not embed a per-run machine fingerprint, so runtime inference remains scoped to the documented workstation/protocol.

## Rebuild the submission-primary artifacts

Use the analysis-only finalizer; it does **not** launch a new optimizer campaign:

```bash
python scripts/finalize_submission_state.py
```

Equivalent explicit commands:

```bash
export SWEVO_GENERATED_DIR=generated_submission

python scripts/validate_master_runs.py generated_submission/master_runs.csv
python scripts/aggregate_results.py generated_submission/master_runs.csv generated_submission/summary_by_method.csv
python scripts/run_stats.py generated_submission/master_runs.csv generated_submission/master_stats.csv
python scripts/build_latex_tables.py generated_submission/summary_by_method.csv generated_submission/master_stats.csv --output-dir generated_submission
python scripts/build_robustness_evidence.py generated_submission/master_runs.csv --output-dir generated_submission
python scripts/build_claim_macros.py generated_submission/summary_by_method.csv --output generated_submission/claim_macros.tex
python scripts/build_claim_evidence_map.py --summary generated_submission/summary_by_method.csv --stats generated_submission/master_stats.csv --output generated_submission/claim_evidence_map.csv
python scripts/check_submission_gates.py --manifest configs/experiment_manifest_submission_primary.csv
python scripts/write_pipeline_audit.py --manifest configs/experiment_manifest_submission_primary.csv
```

## Non-negotiable submission rules

- Accepted final rows must have zero final violations.
- Strict-duty rows must have zero overtime.
- Manuscript percentages and significance values must come from the submission-primary artifacts.
- A1/A2/A3 historical rows must not be used to claim isolated component effects.
- ALNS_MS, HGS_MS, and ILS_MS must remain in the primary comparator set.
- `compute_wh` must always be labelled as a model-based proxy, not measured energy.
- The 100% final-feasibility result is shared by all five primary methods and must not be used to claim a unique feasibility advantage for an EDE component.
- The package does not claim equal hyperparameter-search budgets or universal superiority over optimally tuned external ALNS/HGS/ILS implementations.
- The 10,800-run public-benchmark evidence must always be described as the overlapping full-horizon `[0,H]` regime special case; it must not be presented as a sequential non-overlapping three-shift experiment.
