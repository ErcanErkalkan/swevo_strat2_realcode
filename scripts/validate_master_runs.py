#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOCKED_RUNTIME_EXCEPTION_IDS = {
    "R1_4_5__S1_balanced__EDE__seed20",
    "R1_4_5__S1_balanced__StdDE__seed20",
    "R1_4_5__S1_balanced__ALNS_MS__seed20",
    "R1_4_5__S1_balanced__HGS_MS__seed20",
    "R1_4_5__S1_balanced__ILS_MS__seed20",
}

REQUIRED = [
    "run_id","accepted_final","strict_duty_final","v_cap_final","v_tw_final","v_shift_final",
    "overtime_sum_final","j_scaled_init","j_scaled_final","compute_wh","imp_per_wh","runtime_sec"
]


def _manifest_caps(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    caps: dict[str, float] = {}
    for row in rows:
        run_id = row.get("run_id")
        if not run_id:
            run_id = f"{row['instance_id']}__{row['scenario_id']}__{row['method_id']}__seed{int(row['seed']):02d}"
        caps[run_id] = float(row["walltime_cap_s"])
    return caps


def main(path_str: str, manifest_str: str | None = None, tolerance_s: float = 10.0, allow_locked_runtime_exception: bool = False) -> int:
    path = Path(path_str)
    manifest = Path(manifest_str) if manifest_str else None
    caps = _manifest_caps(manifest)
    errors: list[str] = []
    accepted_rows = 0
    seen_run_ids: set[str] = set()
    runtime_offenders: list[tuple[int, str, float, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing columns: {missing}")
        for i, row in enumerate(reader, start=2):
            run_id = row["run_id"]
            if run_id in seen_run_ids:
                errors.append(f"line {i}: duplicate run_id={run_id}")
            seen_run_ids.add(run_id)
            acc = int(float(row["accepted_final"]))
            strict = int(float(row["strict_duty_final"]))
            vcap = float(row["v_cap_final"])
            vtw = float(row["v_tw_final"])
            vshift = float(row["v_shift_final"])
            overtime = float(row["overtime_sum_final"])
            j_init = float(row["j_scaled_init"])
            j_final = float(row["j_scaled_final"])
            compute_wh = float(row["compute_wh"])
            imp = float(row["imp_per_wh"])
            runtime = float(row["runtime_sec"])
            if not math.isfinite(runtime) or runtime <= 0:
                errors.append(f"line {i}: runtime_sec must be finite and positive")
            if acc == 1:
                accepted_rows += 1
                if any(abs(v) > 1e-8 for v in (vcap, vtw, vshift)):
                    errors.append(f"line {i}: accepted row has nonzero violation(s)")
            if strict == 1 and not (acc == 1 and abs(overtime) <= 1e-8):
                errors.append(f"line {i}: strict_duty_final inconsistent")
            if j_final > j_init + 1e-6:
                errors.append(f"line {i}: j_scaled_final exceeds j_scaled_init")
            if compute_wh <= 0:
                errors.append(f"line {i}: compute_wh must be positive")
            if imp < 0:
                errors.append(f"line {i}: imp_per_wh must be nonnegative")
            if caps:
                if run_id not in caps:
                    errors.append(f"line {i}: run_id absent from manifest: {run_id}")
                elif math.isfinite(runtime) and runtime > caps[run_id] + tolerance_s:
                    runtime_offenders.append((i, run_id, runtime, caps[run_id]))
    if runtime_offenders:
        offender_ids = {run_id for _, run_id, _, _ in runtime_offenders}
        locked_exception_ok = (
            allow_locked_runtime_exception
            and len(runtime_offenders) == 5
            and offender_ids == LOCKED_RUNTIME_EXCEPTION_IDS
        )
        if not locked_exception_ok:
            for i, run_id, runtime, cap in runtime_offenders:
                errors.append(
                    f"line {i}: runtime cap exceeded: run_id={run_id} runtime_sec={runtime:.6f} "
                    f"cap_s={cap:.6f} tolerance_s={tolerance_s:.3f}"
                )
    if caps:
        missing_results = sorted(set(caps) - seen_run_ids)
        for run_id in missing_results[:200]:
            errors.append(f"manifest run missing from ledger: {run_id}")
    if errors:
        print("VALIDATION FAILED")
        for e in errors[:200]:
            print("-", e)
        return 1
    runtime_note = ""
    if runtime_offenders and allow_locked_runtime_exception:
        runtime_note = "; exact locked-five runtime deviation deferred to sensitivity-closure gate"
    print(f"VALIDATION PASSED: {accepted_rows} accepted rows checked; runtime caps validated={bool(caps)}{runtime_note}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/primary/runs.csv")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--runtime-tolerance-s", type=float, default=10.0)
    parser.add_argument(
        "--allow-locked-runtime-exception",
        action="store_true",
        help=(
            "Permit only the exact five synchronized historical runtime-cap deviations. "
            "This does not prove closure; the submission gate must still verify the regenerated "
            "protocol-sensitivity artifacts and manuscript disclosure."
        ),
    )
    args = parser.parse_args()
    raise SystemExit(main(args.path, args.manifest, args.runtime_tolerance_s, args.allow_locked_runtime_exception))
