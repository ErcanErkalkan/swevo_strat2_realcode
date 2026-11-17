#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

PROJECT_LABEL = "EDE_SUSTAINABLE_VRPTW"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

CONFIG_FILES = [
    "budget_policy.yaml",
    "experiment_manifest_submission_primary.csv",
    "method_registry.yaml",
    "metrics_schema.yaml",
    "runtime_cap_rerun_R1_4_5_S1_seed20.csv",
    "scenario_registry.csv",
    "stats_plan.yaml",
    "benchmark_inventory_submission_primary.csv",
    "operational_shift_stress_locked.csv",
    "solomon_equal_3shift_feasibility_audit.csv",
    "parameter_sensitivity_locked.csv",
    "anytime_profile_locked.csv",
]
DOC_FILES = [
    "EXPERIMENT_SPLITS.md",
    "PERFORMANCE_FIRST_METHOD_NOTES.md",
    "LIVE_ADAPTER_CONTRACT.md",
]
GENERATED_FILES = [
    "master_runs.csv",
    "summary_by_method.csv",
    "summary_by_method_accepted_only.csv",
    "master_stats.csv",
    "master_stats_accepted_only.csv",
    "table_feasibility_summary.tex",
    "table_feasibility_summary_accepted_only.tex",
    "table_friedman_omnibus.tex",
    "table_friedman_omnibus_accepted_only.tex",
    "table_posthoc_holm.tex",
    "table_posthoc_holm_accepted_only.tex",
    "robustness_by_scenario.csv",
    "robustness_by_structure.csv",
    "tier_native_endpoint_reductions.csv",
    "secondary_endpoint_stats.csv",
    "claim_macros.tex",
    "claim_evidence_map.csv",
    "submission_gating_report.csv",
    "submission_gating_report.md",
    "pipeline_audit_report.txt",
    "evidence_provenance.json",
    "SUBMISSION_EVIDENCE_SCOPE.md",
    "protocol_sensitivity.csv",
    "runtime_cap_deviations.csv",
    "block_effect_bootstrap_ci.csv",
    "stability_by_method.csv",
    "historical_variant_directional_diagnostics.csv",
    "strengthening_evidence_provenance.json",
    "operational_shift_stress_summary.csv",
    "table_operational_shift_stress.tex",
    "operational_shift_evidence_provenance.json",
    "parameter_sensitivity_summary.csv",
    "parameter_sensitivity_stats.csv",
    "table_parameter_sensitivity.tex",
    "parameter_sensitivity_evidence_provenance.json",
    "anytime_profile_summary.csv",
    "anytime_profile_auc.csv",
    "anytime_profile_overall.csv",
    "table_anytime_profile.tex",
    "anytime_profile_evidence_provenance.json",
]
MANUSCRIPT_FILES = [
    "manuscript.pdf",
    "build_provenance.json",
    "elsarticle-template-num.tex",
    "cas-refs.bib",
    "elsarticle-num.bst",
    "Highlights.docx",
    "Highlights.txt",
    "graphical_abstract.png",
    "graphical_abstract.pdf",
    "graphical_abstract_source.tex",
    "graphical_abstract_provenance.json",
]
PORTAL_FILES = MANUSCRIPT_FILES + ["COVER_LETTER.txt"]
TOP_LEVEL_FILES = ["README.md", "pyproject.toml", "requirements.txt", ".gitignore"]

SCRIPT_FILES = [
    "acquire_primary_benchmarks.py",
    "aggregate_results.py",
    "build_claim_evidence_map.py",
    "build_claim_macros.py",
    "build_final_release_packages.py",
    "build_latex_tables.py",
    "build_robustness_evidence.py",
    "build_submission_strengthening_evidence.py",
    "build_operational_shift_evidence.py",
    "build_parameter_sensitivity_evidence.py",
    "build_anytime_profile_evidence.py",
    "check_submission_gates.py",
    "compile_manuscript_release.py",
    "compile_graphical_abstract_release.py",
    "finalize_submission_state.py",
    "merge_runtime_cap_rerun.py",
    "run_runtime_cap_rerun_parallel.py",
    "run_stats.py",
    "validate_master_runs.py",
    "write_pipeline_audit.py",
]

TEMPLATE_FILES = [
    "claim_evidence_map_template.csv",
    "master_runs_template.csv",
]

TEST_FILES = [
    "test_benchmark_public_ids.py",
    "test_claim_evidence_map.py",
    "test_ede_config.py",
    "test_finalization_closure.py",
    "test_release_packaging.py",
    "test_runtime_cap_remediation.py",
    "test_stats_views.py",
    "test_submission_gates.py",
    "test_validation.py",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Required release input missing/empty: {path}")


def _read_gate_rows(root: Path) -> list[dict[str, str]]:
    path = root / "generated_submission" / "submission_gating_report.csv"
    _require_file(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_audit(root: Path) -> str:
    path = root / "generated_submission" / "pipeline_audit_report.txt"
    _require_file(path)
    return path.read_text(encoding="utf-8")


def require_closed_state(root: Path) -> None:
    rows = _read_gate_rows(root)
    ids = []
    for row in rows:
        try:
            ids.append(int(float(str(row.get("item_id", "")))))
        except ValueError:
            raise SystemExit("Release blocked: non-numeric gate item_id")
    if len(rows) != 23 or sorted(ids) != list(range(1, 24)):
        raise SystemExit(f"Release blocked: expected exact gate IDs 1..23; found rows={len(rows)} ids={sorted(ids)}")
    bad = [row for row in rows if str(row.get("status", "")).strip().lower() != "pass"]
    if bad:
        raise SystemExit(f"Release blocked: not all 23 gates pass; first bad gate={bad[0]}")
    audit = _read_audit(root)
    required = [
        "- pass=23",
        "- fail=0",
        "- manual_review_needed=0",
        "- gate_exact=1",
        "- submission_primary_closed=1",
    ]
    missing = [token for token in required if token not in audit]
    if missing:
        raise SystemExit(f"Release blocked: audit is not closed; missing {missing}")


def run_preflight(root: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/finalize_submission_state.py", "--validate-only"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/compile_manuscript_release.py", "--verify-only"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/compile_graphical_abstract_release.py", "--verify-only"],
        cwd=root,
        check=True,
    )
    require_closed_state(root)


def _collect_py_tree(root: Path, rel_dir: str) -> list[Path]:
    base = root / rel_dir
    if not base.is_dir():
        raise SystemExit(f"Required release directory missing: {base}")
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def _collect_direct_files(root: Path, rel_dir: str, names: list[str]) -> list[Path]:
    out = []
    for name in names:
        p = root / rel_dir / name
        _require_file(p)
        out.append(p)
    return out


def collect_supplement_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    paths += _collect_direct_files(root, ".", TOP_LEVEL_FILES)
    paths += _collect_direct_files(root, "configs", CONFIG_FILES)
    paths += _collect_direct_files(root, "docs", DOC_FILES)
    paths += _collect_py_tree(root, "src")
    paths += _collect_direct_files(root, "scripts", SCRIPT_FILES)
    paths += _collect_direct_files(root, "tests", TEST_FILES)

    paths += _collect_direct_files(root, "templates", TEMPLATE_FILES)

    paths += _collect_direct_files(root, "generated_submission", GENERATED_FILES)
    paths += _collect_direct_files(root, "manuscript", MANUSCRIPT_FILES)

    # Stable de-duplication while preserving sorted final order.
    unique = {p.resolve(): p for p in paths}
    return sorted(unique.values(), key=lambda p: p.relative_to(root).as_posix())


def collect_portal_paths(root: Path) -> list[Path]:
    return _collect_direct_files(root, "manuscript", PORTAL_FILES)


def _file_map(root: Path, paths: list[Path]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for p in paths:
        rel = p.relative_to(root).as_posix()
        if rel in result:
            raise SystemExit(f"Duplicate release path: {rel}")
        result[rel] = p.read_bytes()
    return result


def _manifest_csv(payload: dict[str, bytes]) -> bytes:
    sio = io.StringIO(newline="")
    w = csv.writer(sio, lineterminator="\n")
    w.writerow(["path", "size_bytes", "sha256"])
    for name in sorted(payload):
        data = payload[name]
        w.writerow([name, len(data), sha256_bytes(data)])
    return sio.getvalue().encode("utf-8")


def _sha_sums(payload: dict[str, bytes]) -> bytes:
    return "".join(f"{sha256_bytes(payload[name])}  {name}\n" for name in sorted(payload)).encode("utf-8")


def _package_status(root: Path) -> bytes:
    evidence = root / "generated_submission" / "evidence_provenance.json"
    build = root / "manuscript" / "build_provenance.json"
    graphical = root / "manuscript" / "graphical_abstract_provenance.json"
    _require_file(evidence)
    _require_file(build)
    _require_file(graphical)
    text = (
        "status=CLOSED\n"
        "submission_gates=23/23\n"
        "submission_primary_closed=1\n"
        f"evidence_provenance_sha256={sha256_file(evidence)}\n"
        f"manuscript_build_provenance_sha256={sha256_file(build)}\n"
        f"graphical_abstract_provenance_sha256={sha256_file(graphical)}\n"
        "benchmark_bytes_embedded=0\n"
        "benchmark_reproduction=official acquisition script + SHA-256 inventory\n"
    )
    return text.encode("utf-8")


def _contract_md(kind: str) -> bytes:
    if kind == "supplement":
        text = """# Final reproducibility supplement contract\n\nThis archive contains the active project-authored source/configuration/test snapshot, the canonical submission-primary ledger and manuscript-facing derived evidence, and manuscript release sources/assets. Third-party public benchmark bytes are intentionally excluded; the package contains the acquisition script and the canonical benchmark SHA-256 inventory needed to verify them. Cache files, Python bytecode, LaTeX build intermediates, historical/development generated outputs, runtime work directories, HOLD markers, and superseded evidence sidecars are excluded.\n"""
    else:
        text = """# Final portal package contract\n\nThis archive contains the portal-facing manuscript, highlights, graphical abstract, cover letter, manuscript source/BibTeX/build provenance, the final reproducibility supplement, and package hashes. It is created only after the submission-primary closure postcondition and evidence/manuscript/graphical-abstract provenance checks pass. Highlights plaintext and Word copies are required to match exactly, the five highlight bullets are limited to 85 characters each, and rerun-sensitive percentages/test statistics are intentionally not duplicated into the cover letter, highlights, or graphical-abstract source; mutable effect values remain authoritative only in regenerated evidence and the synchronized manuscript.\n"""
    return text.encode("utf-8")


def augment_payload(root: Path, payload: dict[str, bytes], kind: str) -> dict[str, bytes]:
    out = dict(payload)
    out["release/PACKAGE_STATUS.txt"] = _package_status(root)
    out["release/PACKAGE_CONTRACT.md"] = _contract_md(kind)
    out["release/FILE_MANIFEST.csv"] = _manifest_csv(out)
    out["release/SHA256SUMS.txt"] = _sha_sums(out)
    return out


def deterministic_zip_bytes(root_name: str, payload: dict[str, bytes]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in sorted(payload):
            arc = PurePosixPath(root_name) / PurePosixPath(rel)
            info = zipfile.ZipInfo(str(arc), date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 & 0xFFFF) << 16
            zf.writestr(info, payload[rel])
    return bio.getvalue()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def build_packages(root: Path, out_dir: Path, label: str) -> dict[str, Path]:
    supplement_payload = augment_payload(root, _file_map(root, collect_supplement_paths(root)), "supplement")
    supplement_name = f"{label}_SUPPLEMENT.zip"
    supplement_bytes = deterministic_zip_bytes(f"{label}_SUPPLEMENT", supplement_payload)
    supplement_path = out_dir / supplement_name
    write_atomic(supplement_path, supplement_bytes)

    portal_payload = _file_map(root, collect_portal_paths(root))
    # Portal consumers should not need to locate the supplement separately.
    portal_payload[supplement_name] = supplement_bytes
    portal_payload[supplement_name + ".sha256"] = (
        f"{sha256_bytes(supplement_bytes)}  {supplement_name}\n".encode("utf-8")
    )
    portal_payload["generated_submission/evidence_provenance.json"] = (
        root / "generated_submission" / "evidence_provenance.json"
    ).read_bytes()
    portal_payload = augment_payload(root, portal_payload, "portal")
    portal_name = f"{label}_PORTAL_PACKAGE.zip"
    portal_bytes = deterministic_zip_bytes(f"{label}_PORTAL", portal_payload)
    portal_path = out_dir / portal_name
    write_atomic(portal_path, portal_bytes)

    sums = (
        f"{sha256_bytes(portal_bytes)}  {portal_name}\n"
        f"{sha256_bytes(supplement_bytes)}  {supplement_name}\n"
    ).encode("utf-8")
    sums_path = out_dir / "SHA256SUMS.txt"
    write_atomic(sums_path, sums)
    return {"portal": portal_path, "supplement": supplement_path, "sums": sums_path}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build deterministic final portal and reproducibility-supplement ZIPs after verified 23/23 submission closure.")
    ap.add_argument("--output-dir", default="release")
    ap.add_argument("--label", default=PROJECT_LABEL)
    ap.add_argument("--check-only", action="store_true", help="Run all release preconditions and validate package inputs without writing ZIPs.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_preflight(root)
    supplement = collect_supplement_paths(root)
    portal = collect_portal_paths(root)
    print(f"release_preflight=PASS supplement_files={len(supplement)} portal_files={len(portal)}", flush=True)
    if args.check_only:
        return 0

    outputs = build_packages(root, (root / args.output_dir).resolve(), args.label)
    for kind, path in outputs.items():
        print(f"{kind}={path.relative_to(root) if path.is_relative_to(root) else path} sha256={sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
