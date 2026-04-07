# Performance-First Method Notes

## Goal

The goal is not merely to run all methods, but to run them in a way that can survive a SWEVO review.

## Performance priorities

1. Keep the **same feasibility layer** for all methods.
2. Match the **evaluation budget**, not only wall time.
3. Log **time-to-first-feasible**, **repair activity**, and **rejections**.
4. Separate:
   - final accepted performance,
   - search diagnostics,
   - post hoc Pareto reporting.
5. Use the same seed list across all methods.

## Recommended implementation order

1. Wire EDE and StdDE first.
2. Lock the decoding / repair / local refinement layer.
3. Add ALNS_MS with the exact same feasibility interface.
4. Add HGS_MS with split decoding but the same acceptance semantics.
5. Add ILS_MS on top of the same move operators.
6. Only after all five are stable, activate A1/A2/A3 ablations.

## Performance traps to avoid

- giving HGS/ALNS richer repair logic than EDE,
- hiding failed runs,
- mixing accepted outputs with diagnostic violation counts,
- comparing methods with different candidate-evaluation counts,
- manually typing percentage claims into the manuscript.

## What counts as “best-performing” here

“Best-performing” must be shown on:
- primary endpoint `j_scaled_final`,
- accepted hard-feasible rate,
- strict-duty compliance,
- compute-side efficiency (`imp_per_wh`),
- post hoc Pareto quality.

No single scalar alone is enough.
