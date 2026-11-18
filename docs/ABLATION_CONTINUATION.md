# Open confirmatory ablation

Current status: 48 valid of 1,080 expected, 1,032 missing, zero invalid/duplicate/unexpected records. Two rows exceed the nominal cap plus 10-second logging tolerance. For C104 / S1_balanced / seed06, all four method records are excluded together in the prespecified sensitivity, leaving 44 current rows. The historical cause remains undetermined. New timing instrumentation cannot retroactively identify it.

The immutable original runner and its source/benchmark locks are preserved with the old relative paths in `archive/history/root_snapshot/`. That is a frozen execution context, not a second active working version. The active wrapper verifies those locks before execution. To inspect the continuation plan without starting an optimizer:

```
python scripts/run_ablation.py --project-root archive/history/root_snapshot --campaign archive/history/root_snapshot/EDE_CONFIRMATORY_ABLATION_20260915 --output results/ablation_execution --dry-run
```

Removing `--dry-run` starts long solver runs. It was not done during this work. The nominal remaining exposure is 1,236 hours (51.5 days) before overhead. Use a dedicated stable machine on AC power, prevent sleep/hibernation/restarts, retain the same locked software and benchmark bytes, and run one optimization process per host. The wrapper uses a host-wide socket guard, atomic per-run checkpoints, UTC timestamps, wall and process CPU timings. Interrupted attempts require an explicit recorded recovery decision; do not silently overwrite them.

After new records, audit with `--campaign data/ablation --execution results/ablation_execution --out results/ablation`. Final analysis must use `scripts/analyze_ablation.py` with the frozen archive campaign and original project context, plus `--execution`; it refuses incomplete or invalid records before the unchanged preregistered analyzer is invoked. Preserve all raw records and the original preregistration. A short pilot cannot substitute for the planned campaign.
