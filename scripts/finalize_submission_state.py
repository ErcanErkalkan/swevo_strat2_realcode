#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV["SWEVO_GENERATED_DIR"] = "generated_submission"
PYTHON = sys.executable
INVENTORY = ROOT / "configs" / "benchmark_inventory_submission_primary.csv"
GATING_REPORT = ROOT / "generated_submission" / "submission_gating_report.csv"
AUDIT_REPORT = ROOT / "generated_submission" / "pipeline_audit_report.txt"
EVIDENCE_PROVENANCE = ROOT / "generated_submission" / "evidence_provenance.json"
MANIFEST = ROOT / "configs" / "experiment_manifest_submission_primary.csv"
CLAIM_TEMPLATE = ROOT / "templates" / "claim_evidence_map_template.csv"
SCOPE_DOC = ROOT / "generated_submission" / "SUBMISSION_EVIDENCE_SCOPE.md"
FULL_HISTORICAL_LEDGER = ROOT / "90_ARCHIVE" / "02_GENERATED_HISTORY" / "generated_real" / "master_runs.csv"
OPERATIONAL_SHIFT_LEDGER = ROOT / "configs" / "operational_shift_stress_locked.csv"
SOLOMON_SHIFT_AUDIT = ROOT / "configs" / "solomon_equal_3shift_feasibility_audit.csv"
PARAMETER_SENSITIVITY_LEDGER = ROOT / "configs" / "parameter_sensitivity_locked.csv"
ANYTIME_PROFILE_LEDGER = ROOT / "configs" / "anytime_profile_locked.csv"

EVIDENCE_OUTPUTS = [
    ROOT / "generated_submission" / "summary_by_method.csv",
    ROOT / "generated_submission" / "summary_by_method_accepted_only.csv",
    ROOT / "generated_submission" / "master_stats.csv",
    ROOT / "generated_submission" / "master_stats_accepted_only.csv",
    ROOT / "generated_submission" / "table_feasibility_summary.tex",
    ROOT / "generated_submission" / "table_feasibility_summary_accepted_only.tex",
    ROOT / "generated_submission" / "table_friedman_omnibus.tex",
    ROOT / "generated_submission" / "table_friedman_omnibus_accepted_only.tex",
    ROOT / "generated_submission" / "table_posthoc_holm.tex",
    ROOT / "generated_submission" / "table_posthoc_holm_accepted_only.tex",
    ROOT / "generated_submission" / "robustness_by_scenario.csv",
    ROOT / "generated_submission" / "robustness_by_structure.csv",
    ROOT / "generated_submission" / "tier_native_endpoint_reductions.csv",
    ROOT / "generated_submission" / "secondary_endpoint_stats.csv",
    ROOT / "generated_submission" / "claim_macros.tex",
    ROOT / "generated_submission" / "claim_evidence_map.csv",
    ROOT / "generated_submission" / "protocol_sensitivity.csv",
    ROOT / "generated_submission" / "runtime_cap_deviations.csv",
    ROOT / "generated_submission" / "block_effect_bootstrap_ci.csv",
    ROOT / "generated_submission" / "stability_by_method.csv",
    ROOT / "generated_submission" / "historical_variant_directional_diagnostics.csv",
    ROOT / "generated_submission" / "strengthening_evidence_provenance.json",
    ROOT / "generated_submission" / "operational_shift_stress_summary.csv",
    ROOT / "generated_submission" / "table_operational_shift_stress.tex",
    ROOT / "generated_submission" / "operational_shift_evidence_provenance.json",
    ROOT / "generated_submission" / "parameter_sensitivity_summary.csv",
    ROOT / "generated_submission" / "parameter_sensitivity_stats.csv",
    ROOT / "generated_submission" / "table_parameter_sensitivity.tex",
    ROOT / "generated_submission" / "parameter_sensitivity_evidence_provenance.json",
    ROOT / "generated_submission" / "anytime_profile_summary.csv",
    ROOT / "generated_submission" / "anytime_profile_auc.csv",
    ROOT / "generated_submission" / "anytime_profile_overall.csv",
    ROOT / "generated_submission" / "table_anytime_profile.tex",
    ROOT / "generated_submission" / "anytime_profile_evidence_provenance.json",
]

EVIDENCE_COMMANDS = [
    [PYTHON, "scripts/validate_master_runs.py", "generated_submission/master_runs.csv", "--manifest", "configs/experiment_manifest_submission_primary.csv", "--runtime-tolerance-s", "10", "--allow-locked-runtime-exception"],
    [PYTHON, "scripts/aggregate_results.py", "generated_submission/master_runs.csv", "generated_submission/summary_by_method.csv"],
    [PYTHON, "scripts/run_stats.py", "generated_submission/master_runs.csv", "generated_submission/master_stats.csv"],
    [PYTHON, "scripts/build_latex_tables.py", "generated_submission/summary_by_method.csv", "generated_submission/master_stats.csv", "--output-dir", "generated_submission"],
    [PYTHON, "scripts/build_robustness_evidence.py", "generated_submission/master_runs.csv", "--output-dir", "generated_submission"],
    [PYTHON, "scripts/build_claim_macros.py", "generated_submission/summary_by_method.csv", "--output", "generated_submission/claim_macros.tex"],
    [PYTHON, "scripts/build_submission_strengthening_evidence.py", "--primary", "generated_submission/master_runs.csv", "--full", "90_ARCHIVE/02_GENERATED_HISTORY/generated_real/master_runs.csv", "--out", "generated_submission"],
    [PYTHON, "scripts/build_operational_shift_evidence.py", "--ledger", "configs/operational_shift_stress_locked.csv", "--solomon-audit", "configs/solomon_equal_3shift_feasibility_audit.csv", "--out", "generated_submission"],
    [PYTHON, "scripts/build_parameter_sensitivity_evidence.py", "--ledger", "configs/parameter_sensitivity_locked.csv", "--out", "generated_submission"],
    [PYTHON, "scripts/build_anytime_profile_evidence.py", "--ledger", "configs/anytime_profile_locked.csv", "--out", "generated_submission"],
    [PYTHON, "scripts/build_claim_evidence_map.py", "--summary", "generated_submission/summary_by_method.csv", "--stats", "generated_submission/master_stats.csv", "--output", "generated_submission/claim_evidence_map.csv"],
]

VALIDATION_COMMANDS = [
    [PYTHON, "scripts/check_submission_gates.py", "--manifest", "configs/experiment_manifest_submission_primary.csv"],
    [PYTHON, "scripts/write_pipeline_audit.py", "--manifest", "configs/experiment_manifest_submission_primary.csv"],
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_benchmark_inventory() -> None:
    if not INVENTORY.exists() or INVENTORY.stat().st_size == 0:
        raise SystemExit(
            "Missing benchmark inventory. Run `python scripts/run_runtime_cap_rerun_parallel.py --dry-run` "
            "on the production checkout before regeneration/final validation."
        )
    with INVENTORY.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {
        "benchmark_family", "customer_count", "instance_id", "relative_path",
        "bytes", "sha256", "official_source_archive",
    }
    fields = set(rows[0].keys()) if rows else set()
    missing = sorted(required - fields)
    if missing:
        raise SystemExit(f"Benchmark inventory is missing required columns: {missing}")
    if len(rows) != 36 or len({str(r["instance_id"]) for r in rows}) != 36:
        raise SystemExit(
            f"Benchmark inventory must contain exactly 36 unique submission-primary instances; found rows={len(rows)} unique={len({str(r['instance_id']) for r in rows})}"
        )
    bad_hash = [r["instance_id"] for r in rows if not re.fullmatch(r"[0-9a-fA-F]{64}", str(r.get("sha256", "")))]
    bad_size = []
    for r in rows:
        try:
            if int(r.get("bytes", "0")) <= 0:
                bad_size.append(r["instance_id"])
        except (TypeError, ValueError):
            bad_size.append(r["instance_id"])
    if bad_hash or bad_size:
        raise SystemExit(f"Benchmark inventory has invalid hashes/sizes: bad_hash={bad_hash[:8]} bad_size={bad_size[:8]}")
    print(f"benchmark_inventory=PASS rows=36 sha256={_sha256(INVENTORY)}", flush=True)


def _run(commands: list[list[str]]) -> None:
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }



def _evidence_code_inputs() -> list[Path]:
    paths: set[Path] = {ROOT / "scripts" / "finalize_submission_state.py"}
    for cmd in EVIDENCE_COMMANDS:
        if len(cmd) > 1 and str(cmd[1]).endswith(".py"):
            paths.add(ROOT / str(cmd[1]))
    src_root = ROOT / "src" / "swevo_suite"
    if not src_root.is_dir():
        raise SystemExit(f"Cannot bind evidence generator code; missing source directory: {src_root}")
    paths.update(p for p in src_root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())

def _write_evidence_provenance() -> None:
    master = ROOT / "generated_submission" / "master_runs.csv"
    inputs = [master, FULL_HISTORICAL_LEDGER, MANIFEST, INVENTORY, CLAIM_TEMPLATE, SCOPE_DOC, OPERATIONAL_SHIFT_LEDGER, SOLOMON_SHIFT_AUDIT, PARAMETER_SENSITIVITY_LEDGER, ANYTIME_PROFILE_LEDGER, *_evidence_code_inputs()]
    required = inputs + EVIDENCE_OUTPUTS
    missing = [p.relative_to(ROOT).as_posix() for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise SystemExit(f"Cannot write evidence provenance; missing/empty artifacts: {missing[:12]}")
    payload = {
        "schema": "ede-submission-evidence-provenance-v2",
        "inputs": [_file_record(path) for path in inputs],
        "outputs": [_file_record(path) for path in EVIDENCE_OUTPUTS],
    }
    EVIDENCE_PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    temp = EVIDENCE_PROVENANCE.with_suffix(EVIDENCE_PROVENANCE.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, EVIDENCE_PROVENANCE)
    print(
        f"evidence_provenance=PASS sha256={_sha256(EVIDENCE_PROVENANCE)} "+
        f"master_sha256={_sha256(master)}",
        flush=True,
    )


def _require_evidence_provenance_current() -> None:
    if not EVIDENCE_PROVENANCE.exists() or EVIDENCE_PROVENANCE.stat().st_size == 0:
        raise SystemExit(
            "Missing evidence_provenance.json. Run `python scripts/finalize_submission_state.py --evidence-only` "
            "after the final master ledger merge and before manuscript synchronization/final validation."
        )
    try:
        payload = json.loads(EVIDENCE_PROVENANCE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Invalid evidence provenance JSON: {exc}") from exc
    if payload.get("schema") != "ede-submission-evidence-provenance-v2":
        raise SystemExit(f"Unsupported evidence provenance schema: {payload.get('schema')!r}")

    records = list(payload.get("inputs", [])) + list(payload.get("outputs", []))
    by_path = {str(record.get("path", "")): record for record in records}
    expected_paths = [
        ROOT / "generated_submission" / "master_runs.csv", FULL_HISTORICAL_LEDGER, MANIFEST, INVENTORY, CLAIM_TEMPLATE, SCOPE_DOC, OPERATIONAL_SHIFT_LEDGER, SOLOMON_SHIFT_AUDIT, PARAMETER_SENSITIVITY_LEDGER, ANYTIME_PROFILE_LEDGER,
        *_evidence_code_inputs(), *EVIDENCE_OUTPUTS
    ]
    problems = []
    for path in expected_paths:
        rel = path.relative_to(ROOT).as_posix()
        record = by_path.get(rel)
        if record is None:
            problems.append(f"unrecorded:{rel}")
            continue
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"missing_or_empty:{rel}")
            continue
        current_hash = _sha256(path)
        if current_hash.lower() != str(record.get("sha256", "")).lower():
            problems.append(f"hash_mismatch:{rel}")
            continue
        if path.stat().st_size != int(record.get("bytes", -1)):
            problems.append(f"size_mismatch:{rel}")
    if problems:
        raise SystemExit(
            "Evidence provenance is stale or incomplete; rerun `finalize_submission_state.py --evidence-only`: "
            + "; ".join(problems[:12])
        )
    print(
        f"evidence_provenance_current=PASS sha256={_sha256(EVIDENCE_PROVENANCE)}",
        flush=True,
    )


def _require_final_closure() -> None:
    if not GATING_REPORT.exists() or GATING_REPORT.stat().st_size == 0:
        raise SystemExit(f"Missing final gating report: {GATING_REPORT.relative_to(ROOT)}")
    with GATING_REPORT.open(newline="", encoding="utf-8-sig") as f:
        gate_rows = list(csv.DictReader(f))
    gate_ids = []
    bad_rows = []
    for row in gate_rows:
        try:
            gate_id = int(float(str(row.get("item_id", ""))))
        except (TypeError, ValueError):
            bad_rows.append(f"invalid_item_id={row.get('item_id')!r}")
            continue
        gate_ids.append(gate_id)
        status = str(row.get("status", "")).strip().lower()
        if status != "pass":
            bad_rows.append(f"gate_{gate_id}={status or 'missing'}")
    if len(gate_rows) != 23 or sorted(gate_ids) != list(range(1, 24)) or bad_rows:
        raise SystemExit(
            "Final gating postcondition failed: "
            f"rows={len(gate_rows)} ids={sorted(gate_ids)} bad={bad_rows[:10]}"
        )

    if not AUDIT_REPORT.exists() or AUDIT_REPORT.stat().st_size == 0:
        raise SystemExit(f"Missing final pipeline audit: {AUDIT_REPORT.relative_to(ROOT)}")
    audit_text = AUDIT_REPORT.read_text(encoding="utf-8", errors="replace")
    required_markers = [
        "- pass=23",
        "- fail=0",
        "- manual_review_needed=0",
        "- gate_exact=1",
        "- submission_primary_closed=1",
    ]
    missing_markers = [marker for marker in required_markers if marker not in audit_text]
    if missing_markers:
        raise SystemExit(f"Final audit postcondition failed; missing markers: {missing_markers}")

    print(
        f"submission_closure=PASS gates=23/23 audit_sha256={_sha256(AUDIT_REPORT)}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate submission-primary evidence and/or run final manuscript-aware validation."
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--evidence-only",
        action="store_true",
        help="Regenerate derived evidence after the locked merge, but do not run manuscript-aware gates yet.",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Run final gates and pipeline audit after manuscript synchronization and PDF rebuild.",
    )
    args = ap.parse_args()

    _require_benchmark_inventory()

    if args.evidence_only:
        _run(EVIDENCE_COMMANDS)
        _write_evidence_provenance()
        print(
            "evidence_regeneration=PASS; next synchronize manuscript quantitative text/tables, "
            "rebuild manuscript PDF, then run finalize_submission_state.py --validate-only",
            flush=True,
        )
        return

    if args.validate_only:
        _require_evidence_provenance_current()
        _run(VALIDATION_COMMANDS)
        _require_final_closure()
        print("final_validation=PASS", flush=True)
        return

    # Backward-compatible full mode. It is appropriate only when the manuscript
    # is already synchronized with the current regenerated evidence.
    _run(EVIDENCE_COMMANDS)
    _write_evidence_provenance()
    _require_evidence_provenance_current()
    _run(VALIDATION_COMMANDS)
    _require_final_closure()
    print("submission_finalization=PASS", flush=True)


if __name__ == "__main__":
    main()
