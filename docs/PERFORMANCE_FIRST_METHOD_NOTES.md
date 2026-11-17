# Performance-First Method Notes

## Submission goal

The final paper is evaluated on the completed **10,800-run five-method primary comparison**, not on a new ablation campaign.

Primary methods:
1. EDE
2. StdDE
3. ALNS_MS
4. HGS_MS
5. ILS_MS

Each method has 2,160 completed runs over 36 instances, three scenarios, and 20 paired seeds.

## Evidence hierarchy

1. **Primary evidence:** integrated-method performance of EDE versus four independent comparator families.
2. **Secondary evidence:** native cost, energy, CO2, runtime, and the model-based compute-energy proxy.
3. **Diagnostic evidence:** time-to-first-feasible, repairs, rejected offspring, archives, and convergence traces.
4. **Code-level component definitions:** A1/A2/A3 are retained for reproducibility but are not used for submission-primary causal attribution.

## Why the historical ablation rows are excluded

A1/A2/A3 were remediated on 2026-09-02 so that each variant removes only one conceptual EDE component family. The completed historical 6,480 ablation rows predate that locked definition. Reinterpreting them as clean single-factor evidence would be methodologically incorrect.

The current submission therefore does not claim:
- that seeding contributes a particular percentage,
- that jDE adaptation contributes a particular percentage,
- that LNS contributes a particular percentage,
- or that one component is statistically more important than another.

No new ablation rerun is required because those component-effect claims are outside the manuscript scope.

## Fairness controls

- common problem construction and final feasibility definitions,
- same scenario instances,
- paired seed lists,
- tier-matched wall-clock caps,
- block-level inference on 108 `instance_id × scenario_id` medians,
- Holm correction for the primary EDE-versus-comparator family.

`eval_budget` remains a nominal outer search-step guard rather than a fully instrumented count of every nested objective evaluation.

## Compute-side reporting

`compute_wh` is a runtime-based proxy using assumed tier power:
- small: 52 W
- medium: 86 W
- large: 128 W

It is not a physical hardware energy measurement. The paper must not claim measured compute-energy savings.

## Quantitative authority

This document intentionally does **not** duplicate final Friedman, Wilcoxon-Holm, Kendall W, tier-median, or percentage-reduction values. Those quantities are regenerated from the authoritative `generated_submission/master_runs.csv` ledger and may change when the locked runtime-cap remediation rows are replaced.

For manuscript or release numbers, use only the current regenerated artifacts under `generated_submission/`, especially `master_stats.csv`, `summary_by_method.csv`, `robustness_by_scenario.csv`, `robustness_by_structure.csv`, `tier_native_endpoint_reductions.csv`, and `claim_evidence_map.csv`. `generated_submission/evidence_provenance.json` must be current before final validation or release packaging.

The stable interpretation remains: the paper reports integrated-system performance for the five-method submission-primary comparison; it does not use the historical A1/A2/A3 rows for isolated component-effect estimates.
