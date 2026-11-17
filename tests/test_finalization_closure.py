from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "finalize_submission_state.py"


def _load_script_module():
    spec = spec_from_file_location("finalize_submission_state", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_gate_report(path: Path, failing_gate: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "title", "status", "notes"])
        writer.writeheader()
        for gate_id in range(1, 24):
            writer.writerow({
                "item_id": gate_id,
                "title": f"gate {gate_id}",
                "status": "fail" if gate_id == failing_gate else "pass",
                "notes": "test",
            })


def _write_audit(path: Path, closed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Submission gates:\n"
        "- pass=23\n"
        "- fail=0\n"
        "- manual_review_needed=0\n"
        "Closure checks:\n"
        "- gate_exact=1\n"
        f"- submission_primary_closed={1 if closed else 0}\n",
        encoding="utf-8",
    )


def test_final_closure_requires_exact_23_passes(tmp_path: Path) -> None:
    module = _load_script_module()
    module.ROOT = tmp_path
    module.GATING_REPORT = tmp_path / "generated_submission" / "submission_gating_report.csv"
    module.AUDIT_REPORT = tmp_path / "generated_submission" / "pipeline_audit_report.txt"
    _write_gate_report(module.GATING_REPORT)
    _write_audit(module.AUDIT_REPORT, closed=True)
    module._require_final_closure()


def test_final_closure_rejects_any_failed_gate(tmp_path: Path) -> None:
    module = _load_script_module()
    module.ROOT = tmp_path
    module.GATING_REPORT = tmp_path / "generated_submission" / "submission_gating_report.csv"
    module.AUDIT_REPORT = tmp_path / "generated_submission" / "pipeline_audit_report.txt"
    _write_gate_report(module.GATING_REPORT, failing_gate=22)
    _write_audit(module.AUDIT_REPORT, closed=True)
    try:
        module._require_final_closure()
    except SystemExit as exc:
        assert "gating postcondition failed" in str(exc).lower()
    else:
        raise AssertionError("Failed gate was incorrectly accepted")


def test_final_closure_rejects_open_audit(tmp_path: Path) -> None:
    module = _load_script_module()
    module.ROOT = tmp_path
    module.GATING_REPORT = tmp_path / "generated_submission" / "submission_gating_report.csv"
    module.AUDIT_REPORT = tmp_path / "generated_submission" / "pipeline_audit_report.txt"
    _write_gate_report(module.GATING_REPORT)
    _write_audit(module.AUDIT_REPORT, closed=False)
    try:
        module._require_final_closure()
    except SystemExit as exc:
        assert "audit postcondition failed" in str(exc).lower()
    else:
        raise AssertionError("Open audit was incorrectly accepted")


def _configure_provenance_fixture(module, tmp_path: Path) -> Path:
    generated = tmp_path / "generated_submission"
    configs = tmp_path / "configs"
    templates = tmp_path / "templates"
    generated.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    templates.mkdir(parents=True, exist_ok=True)
    module.ROOT = tmp_path
    module.INVENTORY = configs / "benchmark_inventory_submission_primary.csv"
    module.MANIFEST = configs / "experiment_manifest_submission_primary.csv"
    module.CLAIM_TEMPLATE = templates / "claim_evidence_map_template.csv"
    module.SCOPE_DOC = generated / "SUBMISSION_EVIDENCE_SCOPE.md"
    module.FULL_HISTORICAL_LEDGER = tmp_path / "90_ARCHIVE" / "02_GENERATED_HISTORY" / "generated_real" / "master_runs.csv"
    module.FULL_HISTORICAL_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    module.OPERATIONAL_SHIFT_LEDGER = configs / "operational_shift_stress_locked.csv"
    module.SOLOMON_SHIFT_AUDIT = configs / "solomon_equal_3shift_feasibility_audit.csv"
    module.PARAMETER_SENSITIVITY_LEDGER = configs / "parameter_sensitivity_locked.csv"
    module.ANYTIME_PROFILE_LEDGER = configs / "anytime_profile_locked.csv"
    module.EVIDENCE_PROVENANCE = generated / "evidence_provenance.json"
    module.EVIDENCE_OUTPUTS = [generated / "summary.csv", generated / "table.tex"]
    code_file = tmp_path / "scripts" / "generator.py"
    code_file.parent.mkdir(parents=True, exist_ok=True)
    code_file.write_text("print('generator')\n", encoding="utf-8")
    module._evidence_code_inputs = lambda: [code_file]
    master = generated / "master_runs.csv"
    for path, text in [
        (module.INVENTORY, "inventory"),
        (module.MANIFEST, "manifest"),
        (module.CLAIM_TEMPLATE, "template"),
        (module.SCOPE_DOC, "stable-scope"),
        (module.FULL_HISTORICAL_LEDGER, "historical-v1"),
        (module.OPERATIONAL_SHIFT_LEDGER, "operational-v1"),
        (module.SOLOMON_SHIFT_AUDIT, "solomon-audit-v1"),
        (module.PARAMETER_SENSITIVITY_LEDGER, "parameter-sensitivity-v1"),
        (module.ANYTIME_PROFILE_LEDGER, "anytime-profile-v1"),
        (master, "master-v1"),
        (module.EVIDENCE_OUTPUTS[0], "summary-v1"),
        (module.EVIDENCE_OUTPUTS[1], "table-v1"),
    ]:
        path.write_text(text, encoding="utf-8")
    return master


def test_evidence_provenance_matches_current_master_and_outputs(tmp_path: Path) -> None:
    module = _load_script_module()
    _configure_provenance_fixture(module, tmp_path)
    module._write_evidence_provenance()
    module._require_evidence_provenance_current()


def test_evidence_provenance_rejects_master_change(tmp_path: Path) -> None:
    module = _load_script_module()
    master = _configure_provenance_fixture(module, tmp_path)
    module._write_evidence_provenance()
    master.write_text("master-v2", encoding="utf-8")
    try:
        module._require_evidence_provenance_current()
    except SystemExit as exc:
        assert "hash_mismatch:generated_submission/master_runs.csv" in str(exc)
    else:
        raise AssertionError("Stale evidence provenance was incorrectly accepted")

def test_evidence_provenance_rejects_generator_code_change(tmp_path: Path) -> None:
    module = _load_script_module()
    _configure_provenance_fixture(module, tmp_path)
    code_file = module._evidence_code_inputs()[0]
    module._write_evidence_provenance()
    code_file.write_text("print('changed')\n", encoding="utf-8")
    try:
        module._require_evidence_provenance_current()
    except SystemExit as exc:
        assert "hash_mismatch:scripts/generator.py" in str(exc)
    else:
        raise AssertionError("Generator-code drift was incorrectly accepted")


def test_evidence_provenance_is_deterministic_for_same_state(tmp_path: Path) -> None:
    module = _load_script_module()
    _configure_provenance_fixture(module, tmp_path)
    module._write_evidence_provenance()
    first = module.EVIDENCE_PROVENANCE.read_bytes()
    module._write_evidence_provenance()
    second = module.EVIDENCE_PROVENANCE.read_bytes()
    assert first == second



def test_evidence_provenance_rejects_scope_contract_change(tmp_path: Path) -> None:
    module = _load_script_module()
    _configure_provenance_fixture(module, tmp_path)
    module._write_evidence_provenance()
    module.SCOPE_DOC.write_text("scope-changed", encoding="utf-8")
    try:
        module._require_evidence_provenance_current()
    except SystemExit as exc:
        assert "hash_mismatch:generated_submission/SUBMISSION_EVIDENCE_SCOPE.md" in str(exc)
    else:
        raise AssertionError("Scope-contract drift was incorrectly accepted")


def test_evidence_provenance_rejects_anytime_profile_change(tmp_path: Path) -> None:
    module = _load_script_module()
    _configure_provenance_fixture(module, tmp_path)
    module._write_evidence_provenance()
    module.ANYTIME_PROFILE_LEDGER.write_text("anytime-profile-v2", encoding="utf-8")
    try:
        module._require_evidence_provenance_current()
    except SystemExit as exc:
        assert "hash_mismatch:configs/anytime_profile_locked.csv" in str(exc)
    else:
        raise AssertionError("Anytime-ledger drift was incorrectly accepted")
