#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from datetime import date
import csv
import pandas as pd

from swevo_suite.paths import CONFIGS, GENERATED

PRIMARY_METHODS = {"EDE", "StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"}


def _safe_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _manifest_count(path: Path) -> tuple[int, set[str]]:
    if not path.exists():
        return 0, set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return len(rows), {str(r.get("method_id", "")).strip() for r in rows if r.get("method_id")}


def _note_has_real_source(series: pd.Series) -> bool:
    if series.empty:
        return False
    return series.fillna("").astype(str).str.contains("problem_source=real", regex=False).all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(GENERATED / "pipeline_audit_report.txt"))
    parser.add_argument("--manifest", default=str(CONFIGS / "experiment_manifest_submission_primary.csv"))
    args = parser.parse_args()

    ledger = _safe_csv(GENERATED / "master_runs.csv")
    stats = _safe_csv(GENERATED / "master_stats.csv")
    claims = _safe_csv(GENERATED / "claim_evidence_map.csv")
    gating = _safe_csv(GENERATED / "submission_gating_report.csv")
    manifest_rows, manifest_methods = _manifest_count(Path(args.manifest))

    ledger_methods = set(ledger["method_id"].dropna().astype(str)) if not ledger.empty and "method_id" in ledger else set()
    rows = len(ledger)
    unique_runs = int(ledger["run_id"].nunique()) if not ledger.empty and "run_id" in ledger else 0
    instances = int(ledger["instance_id"].nunique()) if not ledger.empty and "instance_id" in ledger else 0
    scenarios = int(ledger["scenario_id"].nunique()) if not ledger.empty and "scenario_id" in ledger else 0
    seeds = int(ledger["seed"].nunique()) if not ledger.empty and "seed" in ledger else 0
    duplicate_runs = rows - unique_runs
    accepted_rate = float(ledger["accepted_final"].mean() * 100.0) if not ledger.empty else float("nan")
    strict_rate = float(ledger["strict_duty_final"].mean() * 100.0) if not ledger.empty else float("nan")
    all_real = _note_has_real_source(ledger["notes"]) if not ledger.empty and "notes" in ledger else False

    omnibus = stats[stats.get("section", pd.Series(dtype=str)) == "omnibus"] if not stats.empty else pd.DataFrame()
    blocks = int(omnibus.iloc[0]["blocks"]) if not omnibus.empty else 0
    k_methods = int(omnibus.iloc[0]["k_methods"]) if not omnibus.empty else 0

    supported_claims = int(claims["status"].astype(str).str.startswith("supported").sum()) if not claims.empty and "status" in claims else 0
    total_claims = len(claims)

    gating_pass = int((gating.get("status", pd.Series(dtype=object)) == "pass").sum()) if not gating.empty else 0
    gating_fail = int((gating.get("status", pd.Series(dtype=object)) == "fail").sum()) if not gating.empty else 0
    gating_manual = int((gating.get("status", pd.Series(dtype=object)) == "manual_review_needed").sum()) if not gating.empty else 0

    methods_exact = ledger_methods == PRIMARY_METHODS and manifest_methods == PRIMARY_METHODS
    ledger_exact = rows == 10800 and unique_runs == 10800 and duplicate_runs == 0
    design_exact = instances == 36 and scenarios == 3 and seeds == 20
    stats_exact = blocks == 108 and k_methods == 5
    gate_ids = set(pd.to_numeric(gating.get("item_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)) if not gating.empty else set()
    gate_exact = gate_ids == set(range(1, 24)) and gating_pass == 23 and gating_fail == 0 and gating_manual == 0
    claims_exact = total_claims >= 7 and supported_claims == total_claims

    closure = all([methods_exact, ledger_exact, design_exact, all_real, stats_exact, gate_exact, claims_exact])

    lines = [
        "SWEVO submission-primary pipeline audit",
        f"date={date.today().isoformat()}",
        "",
        "Locked primary design:",
        f"- manifest_rows={manifest_rows}",
        f"- manifest_methods={','.join(sorted(manifest_methods))}",
        f"- ledger_rows={rows}",
        f"- unique_run_ids={unique_runs}",
        f"- duplicate_run_ids={duplicate_runs}",
        f"- ledger_methods={','.join(sorted(ledger_methods))}",
        f"- instances={instances}",
        f"- scenarios={scenarios}",
        f"- seeds={seeds}",
        f"- all_problem_sources_real={int(all_real)}",
        "",
        "Feasibility:",
        f"- accepted_rate_pct={accepted_rate:.6f}" if accepted_rate == accepted_rate else "- accepted_rate_pct=NA",
        f"- strict_duty_rate_pct={strict_rate:.6f}" if strict_rate == strict_rate else "- strict_duty_rate_pct=NA",
        "",
        "Inference:",
        f"- independent_blocks={blocks}",
        f"- k_methods={k_methods}",
        "- block_unit=instance_id x scenario_id (20 seeds collapsed within block)",
        "",
        "Claim evidence:",
        f"- supported_claims={supported_claims}",
        f"- total_claims={total_claims}",
        "",
        "Submission gates:",
        f"- pass={gating_pass}",
        f"- fail={gating_fail}",
        f"- manual_review_needed={gating_manual}",
        "",
        "Closure checks:",
        f"- methods_exact={int(methods_exact)}",
        f"- ledger_exact={int(ledger_exact)}",
        f"- design_exact={int(design_exact)}",
        f"- stats_exact={int(stats_exact)}",
        f"- claims_exact={int(claims_exact)}",
        f"- gate_exact={int(gate_exact)}",
        f"- submission_primary_closed={int(closure)}",
        "",
        "Scope safeguards:",
        "- historical A1/A2/A3 rows are excluded from primary inference",
        "- wall-clock-by-tier is the fairness currency; eval_budget is only an outer-step guard",
        "- compute_wh is a model-based runtime proxy, not hardware-metered electrical energy",
        "- the public-benchmark evidence uses overlapping full-horizon operating regimes, not sequential non-overlapping driver shifts",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
