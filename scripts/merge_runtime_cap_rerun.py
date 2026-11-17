#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import pandas as pd

EXPECTED = {
    "R1_4_5__S1_balanced__EDE__seed20",
    "R1_4_5__S1_balanced__StdDE__seed20",
    "R1_4_5__S1_balanced__ALNS_MS__seed20",
    "R1_4_5__S1_balanced__HGS_MS__seed20",
    "R1_4_5__S1_balanced__ILS_MS__seed20",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Safely replace only the five runtime-cap-remediation rows.")
    ap.add_argument("--master", default="generated_submission/master_runs.csv")
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--output", default="generated_submission/master_runs.csv")
    ap.add_argument("--cap-s", type=float, default=7200.0)
    ap.add_argument("--tolerance-s", type=float, default=10.0)
    args = ap.parse_args()

    master = pd.read_csv(args.master)
    rerun = pd.read_csv(args.rerun)

    required_rerun = {"run_id", "runtime_sec", "accepted_final", "strict_duty_final"}
    missing_required = sorted(required_rerun - set(rerun.columns))
    if missing_required:
        raise SystemExit(f"Rerun output missing required columns: {missing_required}")
    if "run_id" not in master.columns:
        raise SystemExit("Master ledger is missing required column: run_id")
    if master["run_id"].duplicated().any():
        raise SystemExit("Master ledger contains duplicate run_id values before merge")

    ids = set(rerun["run_id"].astype(str))
    if ids != EXPECTED or len(rerun) != 5:
        raise SystemExit(f"Rerun file must contain exactly the five locked IDs; found {sorted(ids)}")
    if rerun["run_id"].duplicated().any():
        raise SystemExit("Duplicate run_id in rerun file")

    runtime = pd.to_numeric(rerun["runtime_sec"], errors="coerce")
    invalid_runtime = runtime.isna() | ~runtime.map(math.isfinite) | (runtime <= 0)
    if invalid_runtime.any():
        bad = rerun.loc[invalid_runtime, ["run_id", "runtime_sec"]]
        raise SystemExit(f"Rerun contains missing/non-finite/non-positive runtime_sec:\n{bad.to_string(index=False)}")
    over_cap = runtime > args.cap_s + args.tolerance_s
    if over_cap.any():
        bad = rerun.loc[over_cap, ["run_id", "runtime_sec"]]
        raise SystemExit(f"Rerun still exceeds runtime cap:\n{bad.to_string(index=False)}")

    accepted = pd.to_numeric(rerun["accepted_final"], errors="coerce")
    strict = pd.to_numeric(rerun["strict_duty_final"], errors="coerce")
    if accepted.isna().any() or strict.isna().any() or (accepted != 1).any() or (strict != 1).any():
        raise SystemExit("All five replacement rows must be accepted and strict-duty clean")
    if not EXPECTED.issubset(set(master["run_id"].astype(str))):
        raise SystemExit("Master ledger is missing one or more expected replacement IDs")

    cols = list(master.columns)
    missing_cols = [c for c in cols if c not in rerun.columns]
    if missing_cols:
        raise SystemExit(f"Rerun output missing master columns: {missing_cols}")
    replacement = rerun[cols].copy()
    kept = master[~master["run_id"].astype(str).isin(EXPECTED)].copy()
    merged = pd.concat([kept, replacement], ignore_index=True)
    if len(merged) != len(master) or merged["run_id"].duplicated().any():
        raise SystemExit("Merge would break ledger cardinality or run_id uniqueness")
    # Preserve original canonical ordering by the master run_id order.
    order = {rid: i for i, rid in enumerate(master["run_id"].astype(str))}
    merged["__order"] = merged["run_id"].astype(str).map(order)
    merged = merged.sort_values("__order").drop(columns="__order")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        merged.to_csv(tmp, index=False)
        check = pd.read_csv(tmp)
        if len(check) != len(master) or check["run_id"].duplicated().any():
            raise SystemExit("Atomic merge verification failed before replacing the master ledger")
        if set(check["run_id"].astype(str)) != set(master["run_id"].astype(str)):
            raise SystemExit("Atomic merge verification changed the canonical run_id set")
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"Replaced exactly 5 rows atomically; wrote {output}")


if __name__ == "__main__":
    main()
