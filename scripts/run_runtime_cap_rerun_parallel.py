#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_IDS = {
    "R1_4_5__S1_balanced__EDE__seed20",
    "R1_4_5__S1_balanced__StdDE__seed20",
    "R1_4_5__S1_balanced__ALNS_MS__seed20",
    "R1_4_5__S1_balanced__HGS_MS__seed20",
    "R1_4_5__S1_balanced__ILS_MS__seed20",
}
EXPECTED_METHODS = {"EDE", "StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"}
ENV_OVERRIDES_TO_CLEAR = {
    "SWEVO_EDE_ROUTE_ENDGAME_RESERVE_S",
    "SWEVO_EDE_TRAJECTORY_FRACTION",
    "SWEVO_EDE_ROUTE_ENDGAME_BURST_ITERS",
}


REMEDIATION_DIR = "EDE_RUNTIME_REMEDIATION_2026-09-03"
DEFAULT_OUTPUT = f"{REMEDIATION_DIR}/runtime_cap_rerun_R1_4_5_S1_seed20_results.csv"
DEFAULT_WORK_DIR = f"{REMEDIATION_DIR}/work"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_real_benchmark_inventory(root: Path) -> tuple[Path, str]:
    """Fail fast unless all 36 submission-primary benchmark files are locally present.

    The acquisition helper is invoked in verify-only mode, so this performs no
    network download. It also refreshes the canonical benchmark inventory with
    byte sizes and SHA-256 hashes.
    """
    script = root / "scripts" / "acquire_primary_benchmarks.py"
    if not script.exists():
        raise SystemExit(f"Missing benchmark verification helper: {script}")
    proc = subprocess.run(
        [sys.executable, str(script), "--verify-only"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        details = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
        raise SystemExit(
            "Real-benchmark preflight failed; optimizer rerun was not started."
            + ("\n" + details if details else "")
        )
    inventory = root / "configs" / "benchmark_inventory_submission_primary.csv"
    if not inventory.exists() or inventory.stat().st_size == 0:
        raise SystemExit(f"Benchmark verification did not produce inventory: {inventory}")
    return inventory, _sha256(inventory)


def _run_id(row: dict[str, str]) -> str:
    if row.get("run_id"):
        return str(row["run_id"])
    return (
        f"{row['instance_id']}__{row['scenario_id']}__{row['method_id']}"
        f"__seed{int(row['seed']):02d}"
    )


def _load_locked_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if len(rows) != 5:
        raise SystemExit(f"Locked rerun manifest must contain exactly 5 rows; found {len(rows)}")
    ids = {_run_id(r) for r in rows}
    methods = {str(r.get("method_id", "")) for r in rows}
    if ids != EXPECTED_IDS:
        raise SystemExit(f"Locked rerun IDs differ from expected set: {sorted(ids)}")
    if methods != EXPECTED_METHODS:
        raise SystemExit(f"Locked rerun method set differs from expected set: {sorted(methods)}")
    for row in rows:
        if int(float(row["seed"])) != 20:
            raise SystemExit("Every locked rerun row must use seed 20")
        if str(row["instance_id"]) != "R1_4_5" or str(row["scenario_id"]) != "S1_balanced":
            raise SystemExit("Locked rerun scope must remain R1_4_5 / S1_balanced")
        if abs(float(row["walltime_cap_s"]) - 7200.0) > 1e-9:
            raise SystemExit("Every locked rerun row must retain walltime_cap_s=7200")
    return rows, fieldnames


class _SleepGuard:
    """Prevent Windows system sleep while the matched rerun is active."""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self.enabled = False

    def __enter__(self):
        if os.name == "nt":
            import ctypes

            result = ctypes.windll.kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
            )
            if result == 0:
                raise OSError("SetThreadExecutionState failed; refusing unattended rerun")
            self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        return False


def _write_one_row_manifest(
    path: Path, row: dict[str, str], fieldnames: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the five locked runtime-cap remediation rows in parallel."
    )
    ap.add_argument(
        "--manifest",
        default="configs/runtime_cap_rerun_R1_4_5_S1_seed20.csv",
    )
    ap.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )
    ap.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
    )
    ap.add_argument(
        "--grace-s",
        type=float,
        default=30.0,
        help="External watchdog grace after the 7200 s search cap.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = (root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    work_dir = (root / args.work_dir).resolve() if not Path(args.work_dir).is_absolute() else Path(args.work_dir)

    rows, fieldnames = _load_locked_manifest(manifest)
    ordered_ids = [_run_id(r) for r in rows]
    manifest_hash = _sha256(manifest)
    inventory, inventory_hash = _verify_real_benchmark_inventory(root)

    print("Locked runtime-cap remediation plan")
    print(f"manifest={manifest}")
    print(f"manifest_sha256={manifest_hash}")
    for row in rows:
        print(
            f"  {_run_id(row)} method={row['method_id']} cap={float(row['walltime_cap_s']):.0f}s"
        )
    print(f"benchmark_inventory={inventory}")
    print(f"benchmark_inventory_sha256={inventory_hash}")
    print("benchmark_preflight=PASS")
    if args.dry_run:
        print("dry_run=PASS")
        return 0

    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SWEVO_REQUIRE_REAL_BENCHMARKS"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONHASHSEED"] = "0"
    for key in ENV_OVERRIDES_TO_CLEAR:
        env.pop(key, None)

    processes: dict[str, dict[str, object]] = {}
    started_utc = _utc_now()

    with _SleepGuard():
        for row in rows:
            rid = _run_id(row)
            method = row["method_id"]
            one_manifest = work_dir / f"{method}_manifest.csv"
            one_output = work_dir / f"{method}_result.csv"
            log_path = work_dir / f"{method}.log"
            _write_one_row_manifest(one_manifest, row, fieldnames)
            if one_output.exists():
                one_output.unlink()

            log_file = log_path.open("w", encoding="utf-8")
            cmd = [
                sys.executable,
                "-m",
                "swevo_suite.cli",
                "run",
                "--manifest",
                str(one_manifest),
                "--output",
                str(one_output),
                "--overwrite",
                "--progress",
                "--fail-fast",
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes[rid] = {
                "method": method,
                "proc": proc,
                "log_file": log_file,
                "output": one_output,
                "log": log_path,
                "start": time.perf_counter(),
                "cap": float(row["walltime_cap_s"]),
            }
            print(f"started {rid} pid={proc.pid}")

        failed: list[str] = []
        while processes:
            time.sleep(2.0)
            for rid in list(processes):
                meta = processes[rid]
                proc = meta["proc"]
                elapsed = time.perf_counter() - float(meta["start"])
                cap = float(meta["cap"])
                if proc.poll() is None and elapsed > cap + float(args.grace_s):
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    failed.append(
                        f"{rid}: external watchdog terminated process after {elapsed:.1f}s "
                        f"(cap={cap:.0f}s, grace={args.grace_s:.0f}s)"
                    )
                if proc.poll() is not None:
                    meta["log_file"].close()
                    rc = int(proc.returncode)
                    if rc != 0 and not any(x.startswith(rid + ":") for x in failed):
                        failed.append(f"{rid}: child returned {rc}; see {meta['log']}")
                    del processes[rid]

        if failed:
            raise SystemExit("Matched rerun failed; master ledger was not touched:\n" + "\n".join(failed))

    result_rows: list[dict[str, str]] = []
    output_fieldnames: list[str] | None = None
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        method = row["method_id"]
        path = work_dir / f"{method}_result.csv"
        if not path.exists():
            raise SystemExit(f"Missing child output: {path}")
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            child_rows = list(reader)
            child_fields = list(reader.fieldnames or [])
        if len(child_rows) != 1:
            raise SystemExit(f"{path} must contain exactly one result row")
        rid = str(child_rows[0].get("run_id", ""))
        if rid != _run_id(row):
            raise SystemExit(f"{path} run_id mismatch: {rid}")
        runtime = float(child_rows[0]["runtime_sec"])
        if runtime > float(row["walltime_cap_s"]) + float(args.grace_s):
            raise SystemExit(f"{rid} still violates cap after rerun: runtime={runtime:.3f}s")
        by_id[rid] = child_rows[0]
        if output_fieldnames is None:
            output_fieldnames = child_fields
        elif child_fields != output_fieldnames:
            raise SystemExit("Child outputs have inconsistent schemas")

    result_rows = [by_id[rid] for rid in ordered_ids]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames or [])
        writer.writeheader()
        writer.writerows(result_rows)

    provenance = {
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "benchmark_inventory": str(inventory),
        "benchmark_inventory_sha256": inventory_hash,
        "output": str(output),
        "output_sha256": _sha256(output),
        "run_ids": ordered_ids,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "sleep_prevention": "SetThreadExecutionState" if os.name == "nt" else "not_applicable_non_windows",
        "external_watchdog_grace_s": float(args.grace_s),
        "cleared_environment_overrides": sorted(ENV_OVERRIDES_TO_CLEAR),
    }
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"wrote={output}")
    print(f"provenance={provenance_path}")
    print("matched_rerun=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
