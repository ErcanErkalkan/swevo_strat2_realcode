#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import csv

import pandas as pd

from swevo_suite.paths import CONFIGS, GENERATED, TEMPLATES


def _manual(item_id: int, title: str, notes: str) -> dict[str, object]:
    return {"item_id": item_id, "title": title, "status": "manual_review_needed", "notes": notes}


def _pass(item_id: int, title: str, notes: str) -> dict[str, object]:
    return {"item_id": item_id, "title": title, "status": "pass", "notes": notes}


def _fail(item_id: int, title: str, notes: str) -> dict[str, object]:
    return {"item_id": item_id, "title": title, "status": "fail", "notes": notes}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _manifest_methods(path: Path) -> list[str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted({row["method_id"] for row in rows})


def build_gate_rows(manifest: Path, readme: Path) -> list[dict[str, object]]:
    master_runs = _read_csv(GENERATED / "master_runs.csv")
    summary = _read_csv(GENERATED / "summary_by_method.csv")
    summary_acc = _read_csv(GENERATED / "summary_by_method_accepted_only.csv")
    stats = _read_csv(GENERATED / "master_stats.csv")
    stats_acc = _read_csv(GENERATED / "master_stats_accepted_only.csv")
    claim_map = _read_csv(GENERATED / "claim_evidence_map.csv")
    template = _read_csv(TEMPLATES / "claim_evidence_map_template.csv")

    rows: list[dict[str, object]] = []
    rows.append(_manual(1, "Abstract is filled", "Manuscript source is not available in this repo."))
    rows.append(_manual(2, "Highlights are filled", "Manuscript source is not available in this repo."))
    rows.append(_manual(3, "Comparator list in manuscript == comparator list in manifest", "Manifest methods can be checked locally, but manuscript text must be verified manually."))

    manifest_methods = _manifest_methods(manifest)
    output_methods = sorted(master_runs["method_id"].dropna().unique().tolist()) if not master_runs.empty else []
    missing_methods = [method for method in manifest_methods if method not in output_methods]
    if missing_methods:
        rows.append(_fail(4, "Every comparator has real run outputs in generated/master_runs.csv", f"Missing methods: {', '.join(missing_methods)}"))
    else:
        rows.append(_pass(4, "Every comparator has real run outputs in generated/master_runs.csv", f"Methods present: {', '.join(output_methods)}; rows={len(master_runs)}"))

    accepted_bad = 0
    strict_bad = 0
    if not master_runs.empty:
        accepted_bad = int(((master_runs["accepted_final"] == 1) & ((master_runs["v_cap_final"] != 0) | (master_runs["v_tw_final"] != 0) | (master_runs["v_shift_final"] != 0))).sum())
        strict_bad = int(((master_runs["strict_duty_final"] == 1) & (master_runs["overtime_sum_final"] != 0)).sum())
    rows.append(
        _pass(5, "Accepted rows have zero final violations", "All accepted rows are clean.")
        if accepted_bad == 0
        else _fail(5, "Accepted rows have zero final violations", f"Found {accepted_bad} accepted rows with residual violations.")
    )
    rows.append(
        _pass(6, "Strict-duty rows have zero overtime", "All strict-duty rows have zero overtime.")
        if strict_bad == 0
        else _fail(6, "Strict-duty rows have zero overtime", f"Found {strict_bad} strict-duty rows with overtime.")
    )

    search_tables = [
        GENERATED / "table_feasibility_summary.tex",
        GENERATED / "table_friedman_omnibus.tex",
        GENERATED / "table_posthoc_holm.tex",
    ]
    missing_search_tables = [path.name for path in search_tables if not path.exists()]
    rows.append(
        _pass(7, "Search diagnostics are reported in separate tables", "Separate LaTeX tables are present.")
        if not missing_search_tables
        else _fail(7, "Search diagnostics are reported in separate tables", f"Missing tables: {', '.join(missing_search_tables)}")
    )

    rows.append(
        _pass(8, "Summary tables are auto-generated from master_runs.csv", "summary_by_method.csv is present.")
        if not summary.empty
        else _fail(8, "Summary tables are auto-generated from master_runs.csv", "generated/summary_by_method.csv is missing.")
    )
    rows.append(
        _pass(9, "Stats tables are auto-generated from master_stats.csv", "master_stats.csv is present.")
        if not stats.empty
        else _fail(9, "Stats tables are auto-generated from master_stats.csv", "generated/master_stats.csv is missing.")
    )
    rows.append(
        _pass(10, "Claim macros are generated from the outputs, not typed manually", "generated/claim_macros.tex is present.")
        if (GENERATED / "claim_macros.tex").exists()
        else _fail(10, "Claim macros are generated from the outputs, not typed manually", "generated/claim_macros.tex is missing.")
    )

    required_claims = sorted(template["claim_id"].dropna().astype(str).str.strip().str.upper().tolist()) if not template.empty else []
    claim_rows = claim_map.copy()
    if not claim_rows.empty:
        claim_rows["claim_id"] = claim_rows["claim_id"].astype(str).str.strip().str.upper()
    claim_status = {row["claim_id"]: row["status"] for _, row in claim_rows.iterrows()} if not claim_rows.empty else {}
    missing_claims = [claim_id for claim_id in required_claims if claim_id not in claim_status]
    weak_claims = [claim_id for claim_id in required_claims if claim_status.get(claim_id) in {"todo", "missing_artifacts", "manual_review_needed"}]
    if missing_claims or weak_claims:
        notes = []
        if missing_claims:
            notes.append(f"Missing claim ids: {', '.join(missing_claims)}")
        if weak_claims:
            notes.append(f"Needs work: {', '.join(weak_claims)}")
        rows.append(_fail(11, "Every major claim appears in the claim-evidence map", "; ".join(notes)))
    else:
        rows.append(_pass(11, "Every major claim appears in the claim-evidence map", "All template claim ids are covered by generated claim_evidence_map.csv."))

    readme_text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    required_commands = [
        "python scripts/check_benchmark_inventory.py",
        "python scripts/aggregate_results.py",
        "python scripts/run_stats.py",
        "python scripts/build_latex_tables.py",
        "python scripts/build_claim_macros.py",
        "python scripts/build_claim_evidence_map.py",
    ]
    missing_commands = [cmd for cmd in required_commands if cmd not in readme_text]
    rows.append(
        _pass(12, "README reproduces the full run pipeline", "README contains the core pipeline commands.")
        if not missing_commands
        else _fail(12, "README reproduces the full run pipeline", f"README is missing commands: {', '.join(missing_commands)}")
    )

    split_doc = ROOT / "docs" / "EXPERIMENT_SPLITS.md"
    split_text = split_doc.read_text(encoding="utf-8") if split_doc.exists() else ""
    split_keywords = ["dev tuning split", "medium comparator split", "paper preflight subset", "submission preflight subset"]
    missing_split_keywords = [kw for kw in split_keywords if kw not in split_text.lower()]
    rows.append(
        _pass(13, "Tuning and test splits are fixed and documented", "docs/EXPERIMENT_SPLITS.md documents the fixed split roles.")
        if split_doc.exists() and not missing_split_keywords
        else _manual(13, "Tuning and test splits are fixed and documented", "Split documentation is still incomplete or missing.")
    )

    failure_files = [
        GENERATED / "master_runs_medium_pilot_failures.txt",
        GENERATED / "master_runs_medium_compare_failures.txt",
    ]
    rows.append(
        _pass(14, "Failed runs are reported, not hidden", "Failure-report files are present in generated/.")
        if all(path.exists() for path in failure_files)
        else _fail(14, "Failed runs are reported, not hidden", "Expected failure-report artifacts are missing.")
    )

    cond_ok = not summary.empty and not summary_acc.empty and not stats.empty and not stats_acc.empty
    rows.append(
        _pass(15, "Conditional-on-accepted and unconditional summaries are both available", "Both unconditional and accepted-only summaries/stats are present.")
        if cond_ok
        else _fail(15, "Conditional-on-accepted and unconditional summaries are both available", "Accepted-only and unconditional artifact pairs are incomplete.")
    )

    return rows


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    status_counts = pd.DataFrame(rows)["status"].value_counts().to_dict()
    lines = [
        "# Submission Gating Report",
        "",
        f"- pass: {status_counts.get('pass', 0)}",
        f"- fail: {status_counts.get('fail', 0)}",
        f"- manual_review_needed: {status_counts.get('manual_review_needed', 0)}",
        "",
        "| Item | Status | Notes |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {int(row['item_id'])}. {row['title']} | {row['status']} | {row['notes']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(CONFIGS / "experiment_manifest_full.csv"))
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    parser.add_argument("--csv-output", default=str(GENERATED / "submission_gating_report.csv"))
    parser.add_argument("--md-output", default=str(GENERATED / "submission_gating_report.md"))
    args = parser.parse_args()

    rows = build_gate_rows(Path(args.manifest), Path(args.readme))
    csv_path = Path(args.csv_output)
    md_path = Path(args.md_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_markdown(md_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
