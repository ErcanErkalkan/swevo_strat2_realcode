# Project overview

This project evaluates an integrated feasibility-first Enhanced Differential Evolution architecture for sustainability-aware VRPTW with shift-indexed operating regimes. The completed public-benchmark experiment uses three overlapping full-horizon regimes, not sequential physical driver shifts.

The authoritative primary ledger is `data/primary/runs.csv`; the locked design is `configs/experiment_manifest_submission_primary.csv`. Inputs are immutable evidence. Generated evidence belongs in `results/`; the current article is in `manuscript/`, and the current delivery is in `submission/`. Historic snapshots belong exclusively in `archive/history/`.

The primary comparison contains 10,800 runs over 36 instances, three scenarios, five frozen internal implementations and 20 paired seeds. Secondary suites contain 18 synthetic shift-stress runs, 45 local one-factor parameter runs, and 48 short-budget anytime runs. They have narrower scope than the primary comparison. The confirmatory ablation remains open: 48 valid records, 1,032 missing, two runtime deviations. No new optimization experiment was run during consolidation.

The result supports lower composite model scores for the complete E-DE system among feasible solutions under the documented design. It supports neither a unique feasibility advantage nor physical energy savings, a universal optimizer-family ranking, general convergence claims, or isolated component effects. Packaging readiness and scientific completion of the ablation are separate states.
