#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED_REAL = ROOT / "generated_real"
BATCH_DIR = ROOT / "configs" / "batches"

DEFAULT_METHODS = [
    "EDE",
    "StdDE",
    "ALNS_MS",
    "HGS_MS",
    "ILS_MS",
    "A1_NoSeed",
    "A2_NoJDE",
    "A3_NoLNS",
]


def run_id_from_manifest_row(row: dict[str, str]) -> str:
    return "__".join(
        [
            row["instance_id"],
            row["scenario_id"],
            row["method_id"],
            f"seed{int(row['seed']):02d}",
        ]
    )


def read_done_run_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id")
            if run_id:
                done.add(run_id)
    return done


def write_pending_manifest(method: str, out_dir: Path) -> tuple[Path, int]:
    manifest = BATCH_DIR / f"manifest_{method}.csv"
    output = GENERATED_REAL / f"batch_{method}.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    done = read_done_run_ids(output)
    pending_rows: list[dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Missing header in {manifest}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            if run_id_from_manifest_row(row) not in done:
                pending_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    pending_path = out_dir / f"pending_{method}.csv"
    with pending_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pending_rows)
    return pending_path, len(pending_rows)


def write_missing_report(methods: list[str], path: Path) -> int:
    rows: list[dict[str, str]] = []
    for method in methods:
        manifest = BATCH_DIR / f"manifest_{method}.csv"
        output = GENERATED_REAL / f"batch_{method}.csv"
        done = read_done_run_ids(output)
        with manifest.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = run_id_from_manifest_row(row)
                if run_id not in done:
                    rows.append(
                        {
                            "run_id": run_id,
                            "method_id": row["method_id"],
                            "instance_id": row["instance_id"],
                            "scenario_id": row["scenario_id"],
                            "seed": row["seed"],
                            "tier": row["tier"],
                            "manifest": str(manifest.relative_to(ROOT)),
                        }
                    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_id", "method_id", "instance_id", "scenario_id", "seed", "tier", "manifest"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def run_checked(cmd: list[str], log) -> None:
    print(f"RUN {' '.join(cmd)}", file=log, flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True, stdout=log, stderr=subprocess.STDOUT)


def finalize_outputs(methods: list[str], log) -> None:
    sources = [str(GENERATED_REAL / f"batch_{method}.csv") for method in methods]
    run_checked(
        [
            sys.executable,
            str(ROOT / "scripts" / "combine_master_runs.py"),
            *sources,
            "--output",
            str(GENERATED_REAL / "master_runs.csv"),
        ],
        log,
    )
    run_checked([sys.executable, str(ROOT / "scripts" / "validate_master_runs.py"), str(GENERATED_REAL / "master_runs.csv")], log)
    run_checked(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_results.py"),
            str(GENERATED_REAL / "master_runs.csv"),
            str(GENERATED_REAL / "summary_by_method.csv"),
        ],
        log,
    )
    run_checked(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_stats.py"),
            str(GENERATED_REAL / "master_runs.csv"),
            str(GENERATED_REAL / "master_stats.csv"),
            "--control",
            "EDE",
        ],
        log,
    )
    run_checked(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_latex_tables.py"),
            str(GENERATED_REAL / "summary_by_method.csv"),
            str(GENERATED_REAL / "master_stats.csv"),
            "--output-dir",
            str(GENERATED_REAL),
        ],
        log,
    )
    run_checked(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_claim_macros.py"),
            str(GENERATED_REAL / "summary_by_method.csv"),
            "--output",
            str(GENERATED_REAL / "claim_macros.tex"),
        ],
        log,
    )
    run_checked(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_claim_evidence_map.py"),
            "--summary",
            str(GENERATED_REAL / "summary_by_method.csv"),
            "--stats",
            str(GENERATED_REAL / "master_stats.csv"),
            "--output",
            str(GENERATED_REAL / "claim_evidence_map.csv"),
        ],
        log,
    )


def run_round(methods: list[str], run_dir: Path, round_idx: int, poll_s: int, log) -> int:
    round_dir = run_dir / f"round_{round_idx:03d}"
    manifest_dir = round_dir / "pending_manifests"
    child_log_dir = round_dir / "logs"
    child_log_dir.mkdir(parents=True, exist_ok=True)

    pending: dict[str, tuple[Path, int]] = {}
    for method in methods:
        pending_path, count = write_pending_manifest(method, manifest_dir)
        pending[method] = (pending_path, count)

    total_pending = sum(count for _, count in pending.values())
    print(f"round={round_idx} total_pending={total_pending}", file=log, flush=True)
    for method, (_, count) in pending.items():
        print(f"  {method}: pending={count} current_rows={count_rows(GENERATED_REAL / f'batch_{method}.csv')}", file=log, flush=True)

    if total_pending == 0:
        return 0

    env = os.environ.copy()
    env["SWEVO_REQUIRE_REAL_BENCHMARKS"] = "1"
    env["PYTHONPATH"] = "src"

    procs: list[tuple[str, subprocess.Popen, object, Path]] = []
    pid_rows: list[dict[str, str]] = []
    for method, (manifest, count) in pending.items():
        if count == 0:
            continue
        output = GENERATED_REAL / f"batch_{method}.csv"
        child_log_path = child_log_dir / f"{method}.log"
        child_log = child_log_path.open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_manifest.py"),
            "run",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--progress",
        ]
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=child_log, stderr=subprocess.STDOUT)
        procs.append((method, proc, child_log, child_log_path))
        pid_rows.append(
            {
                "method_id": method,
                "pid": str(proc.pid),
                "pending_rows": str(count),
                "manifest": str(manifest.relative_to(ROOT)),
                "output": str(output.relative_to(ROOT)),
                "log": str(child_log_path.relative_to(ROOT)),
            }
        )
        print(f"started {method} pid={proc.pid} pending={count}", file=log, flush=True)

    with (round_dir / "pids.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method_id", "pid", "pending_rows", "manifest", "output", "log"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pid_rows)

    failures = 0
    while procs:
        time.sleep(poll_s)
        for item in list(procs):
            method, proc, child_log, child_log_path = item
            ret = proc.poll()
            if ret is None:
                continue
            child_log.close()
            procs.remove(item)
            print(f"finished {method} returncode={ret} remaining={len(procs)} log={child_log_path}", file=log, flush=True)
            if ret != 0:
                failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means retry until no missing runs remain.")
    parser.add_argument("--poll-s", type=int, default=30)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_name = datetime.now().strftime("resume_%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else (GENERATED_REAL / "pending_logs" / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "supervisor.log"

    with log_path.open("a", encoding="utf-8") as log:
        print(f"run_dir={run_dir}", file=log, flush=True)
        print(f"methods={','.join(args.methods)}", file=log, flush=True)

        round_idx = 1
        while True:
            failures = run_round(args.methods, run_dir, round_idx, args.poll_s, log)
            missing = write_missing_report(args.methods, GENERATED_REAL / "missing_runs_report.csv")
            print(f"after_round={round_idx} failures={failures} missing={missing}", file=log, flush=True)

            if missing == 0:
                finalize_outputs(args.methods, log)
                print("DONE missing=0", file=log, flush=True)
                return 0

            if args.max_rounds and round_idx >= args.max_rounds:
                print(f"STOP max_rounds={args.max_rounds} missing={missing}", file=log, flush=True)
                return 2
            round_idx += 1


if __name__ == "__main__":
    raise SystemExit(main())
