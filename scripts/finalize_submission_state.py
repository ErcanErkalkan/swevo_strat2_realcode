#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


COMMANDS = [
    ["python", "scripts/check_benchmark_inventory.py"],
    ["python", "scripts/init_run_state.py"],
    ["python", "scripts/prepare_paper_run.py"],
    [
        "python",
        "scripts/prepare_paper_run.py",
        "--methods",
        "EDE",
        "StdDE",
        "ALNS_MS",
        "HGS_MS",
        "ILS_MS",
        "A1_NoSeed",
        "A2_NoJDE",
        "A3_NoLNS",
        "--prefix",
        "submission_full_real",
    ],
    ["python", "scripts/validate_master_runs.py", "generated/master_runs.csv"],
    ["python", "scripts/aggregate_results.py"],
    ["python", "scripts/run_stats.py"],
    ["python", "scripts/build_latex_tables.py"],
    ["python", "scripts/build_claim_macros.py"],
    ["python", "scripts/build_claim_evidence_map.py"],
    ["python", "scripts/write_pipeline_audit.py"],
    ["python", "scripts/check_submission_gates.py"],
]


def main() -> None:
    for cmd in COMMANDS:
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
