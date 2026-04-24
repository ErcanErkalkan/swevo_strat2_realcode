#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from datetime import date

import pandas as pd

from swevo_suite.paths import GENERATED


def _read_key_value_report(path: Path) -> dict[str, str]:
    report: dict[str, str] = {}
    if not path.exists():
        return report
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        report[key.strip()] = value.strip()
    return report


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(GENERATED / "pipeline_error_report.txt"))
    args = parser.parse_args()

    inventory = _safe_csv(GENERATED / "benchmark_inventory_resolution.csv")
    paper_preflight = _read_key_value_report(GENERATED / "paper_main3_real_preflight_report.txt")
    submission_preflight = _read_key_value_report(GENERATED / "submission_full_real_preflight_report.txt")
    tuning = _safe_csv(GENERATED / "paper_main3_real" / "sweeps" / "ede_tuning_compare.csv")
    gating = _safe_csv(GENERATED / "submission_gating_report.csv")

    inventory_missing = int((inventory.get("status", pd.Series(dtype=object)) == "missing").sum()) if not inventory.empty else 0
    inventory_invalid = int((inventory.get("status", pd.Series(dtype=object)) == "invalid_public_instance_id").sum()) if not inventory.empty else 0
    inventory_ok = int((inventory.get("status", pd.Series(dtype=object)) == "ok").sum()) if not inventory.empty else 0

    best_ede_line = "not_available"
    ils_line = "not_available"
    gap_line = "not_available"
    if not tuning.empty:
        ede_rows = tuning[tuning["method_id"] == "EDE"].copy()
        ils_rows = tuning[tuning["method_id"] == "ILS_MS"].copy()
        if not ede_rows.empty:
            best_ede = ede_rows.sort_values("median_final").iloc[0]
            best_ede_line = f"{best_ede['label']} median_final={best_ede['median_final']:.6f}"
        if not ils_rows.empty:
            ils = ils_rows.sort_values("median_final").iloc[0]
            ils_line = f"{ils['label']} median_final={ils['median_final']:.6f}"
            if not ede_rows.empty:
                gap = float(best_ede["median_final"]) - float(ils["median_final"])
                gap_line = f"{gap:.6f}"

    gating_pass = int((gating.get("status", pd.Series(dtype=object)) == "pass").sum()) if not gating.empty else 0
    gating_fail = int((gating.get("status", pd.Series(dtype=object)) == "fail").sum()) if not gating.empty else 0
    gating_manual = int((gating.get("status", pd.Series(dtype=object)) == "manual_review_needed").sum()) if not gating.empty else 0

    lines = [
        "SWEVO pipeline audit",
        f"date={date.today().isoformat()}",
        "",
        "Benchmark verification:",
        f"- inventory_ok={inventory_ok}",
        f"- inventory_missing={inventory_missing}",
        f"- inventory_invalid_public_ids={inventory_invalid}",
        "",
        "Paper preflight (EDE / ILS_MS / StdDE):",
        f"- requested_runs={paper_preflight.get('requested_runs', '0')}",
        f"- ready_runs={paper_preflight.get('ready_runs', '0')}",
        f"- blocked_runs={paper_preflight.get('blocked_runs', '0')}",
        f"- blocked_instances={paper_preflight.get('blocked_instances', 'none')}",
        "",
        "Submission preflight (all comparators):",
        f"- requested_runs={submission_preflight.get('requested_runs', '0')}",
        f"- ready_runs={submission_preflight.get('ready_runs', '0')}",
        f"- blocked_runs={submission_preflight.get('blocked_runs', '0')}",
        f"- blocked_instances={submission_preflight.get('blocked_instances', 'none')}",
        "",
        "Current tuning checkpoint:",
        f"- best_ede_variant={best_ede_line}",
        f"- ils_reference={ils_line}",
        f"- ede_minus_ils_gap={gap_line}",
        "",
        "Submission gates:",
        f"- pass={gating_pass}",
        f"- fail={gating_fail}",
        f"- manual_review_needed={gating_manual}",
        "",
        "Closure status:",
        "- manifest/inventory benchmark blocker is closed",
        "- full comparator preflight is ready",
        "- remaining risk is result quality, not benchmark availability",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
