# SWEVO Strategy-2 Real-Code Experiment Package

This repository hosts the public submission-scope mirror for the manuscript **Feasibility-First Enhanced Differential Evolution for Sustainability-Aware VRPTW with Shift-Indexed Operating Regimes**.

## Submission-primary scope

The final submission-primary evidence set uses five methods:

- EDE
- StdDE
- ALNS_MS
- HGS_MS
- ILS_MS

The authoritative evidence contains 10,800 completed real-benchmark runs:

36 instances × 3 scenarios × 5 methods × 20 paired seeds.

Historical A1_NoSeed, A2_NoJDE, and A3_NoLNS runs are preserved outside the frozen submission tree for provenance but are not used for submission-primary inferential statistics or isolated component-effect claims. A new 6,480-run ablation campaign is not required for the claims made in the current manuscript.

## Temporal/regime semantics

The submission-primary public-benchmark experiment does not partition the Solomon/Homberger planning horizon into three consecutive driver shifts. The loader preserves the native benchmark feasibility region by creating three overlapping shift/regime labels whose availability intervals are all `[0,H]`, where `H` is the original benchmark horizon. Scenarios change operating/emission multipliers; they do not change fleet-share proportions. Vehicle type is selected at route level and no persistent physical vehicle identity is tracked across regime labels.

## Reproducibility evidence

The exact seed-level evidence is distributed with the manuscript as Supplementary Material rather than committed into this Git branch.

Authoritative final archive:

- `SWEVO_FINAL_REPRODUCIBILITY_SUPPLEMENT_2026-09-03.zip`
- SHA-256: `91d495ce7a64bda5b8fd98de0151818a86763e4b5caafaa27f38514d73129ae9`

The public submission-scope branch is `submission-2026-09-02`. The release-style frozen branch ref is `release/v0.2.0-submission-rc1`; it is a branch ref, not a Git tag, because the connected GitHub interface used for the freeze does not expose tag/release creation.

## Primary analysis semantics

- The production comparison is wall-clock matched by tier.
- `eval_budget` is a nominal outer-step guard, not a complete count of every nested objective evaluation.
- `compute_wh` is a model-based runtime × assumed-power proxy, not hardware-metered electrical energy.
- Pairwise Wilcoxon inference collapses the 20 paired seeds to the median within each `instance_id × scenario_id` block.
- Friedman and Wilcoxon-Holm therefore use the same 108 independent blocks.
- The 100% final-feasibility result is shared by all five primary methods and is not evidence of a unique EDE feasibility advantage.

## Rebuild the submission-primary artifacts

Use the analysis-only finalizer; it does not launch a new optimizer campaign:

```bash
python scripts/finalize_submission_state.py
```

The canonical manifest default is `configs/experiment_manifest_submission_primary.csv`.

## Final author declarations

Confirmed on 2026-09-03:

- Funding: no specific grant.
- Competing interests: none.
- CRediT roles: Conceptualization; Methodology; Software; Validation; Formal analysis; Investigation; Data curation; Writing - original draft; Writing - review & editing; Visualization.
- Acknowledgements: none.
- Originality/concurrent submission: the manuscript is not under consideration elsewhere.

## Submission safeguards

- Manuscript percentages and significance values must come from the submission-primary artifacts.
- Historical A1/A2/A3 rows must not be used to claim isolated component effects.
- ALNS_MS, HGS_MS, and ILS_MS remain in the primary comparator set.
- The public-benchmark evidence is the overlapping full-horizon `[0,H]` regime special case; it must not be presented as a sequential non-overlapping three-shift experiment.
- The package does not claim equal hyperparameter-search budgets or universal superiority over optimally tuned external ALNS/HGS/ILS implementations.
