from __future__ import annotations

import csv
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_final_release_packages.py"
spec = spec_from_file_location("build_final_release_packages", SCRIPT)
module = module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def _write(p: Path, data: str = "x\n") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding="utf-8")


def _closed(root: Path) -> None:
    gate = root / "generated_submission/submission_gating_report.csv"
    gate.parent.mkdir(parents=True, exist_ok=True)
    with gate.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "status"])
        w.writeheader()
        for i in range(1, 24):
            w.writerow({"item_id": i, "status": "pass"})
    _write(root / "generated_submission/pipeline_audit_report.txt", "- pass=23\n- fail=0\n- manual_review_needed=0\n- gate_exact=1\n- submission_primary_closed=1\n")


def test_closed_state_requires_exact_23(tmp_path: Path) -> None:
    _closed(tmp_path)
    module.require_closed_state(tmp_path)
    rows = list(csv.DictReader((tmp_path / "generated_submission/submission_gating_report.csv").open()))
    rows[-1]["status"] = "fail"
    with (tmp_path / "generated_submission/submission_gating_report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "status"]); w.writeheader(); w.writerows(rows)
    try:
        module.require_closed_state(tmp_path)
    except SystemExit as exc:
        assert "not all 23 gates pass" in str(exc)
    else:
        raise AssertionError("open gate was accepted")


def test_deterministic_zip_is_byte_reproducible() -> None:
    payload = {"b.txt": b"B", "a.txt": b"A"}
    one = module.deterministic_zip_bytes("ROOT", payload)
    two = module.deterministic_zip_bytes("ROOT", payload)
    assert one == two
    with zipfile.ZipFile(io.BytesIO(one)) as zf:
        assert zf.namelist() == ["ROOT/a.txt", "ROOT/b.txt"]
        assert zf.read("ROOT/a.txt") == b"A"


def test_contract_excludes_cache_and_build_intermediates(tmp_path: Path) -> None:
    # Populate required direct files.
    for name in module.TOP_LEVEL_FILES: _write(tmp_path / name)
    for name in module.CONFIG_FILES: _write(tmp_path / "configs" / name)
    for name in module.DOC_FILES: _write(tmp_path / "docs" / name)
    for name in module.GENERATED_FILES: _write(tmp_path / "generated_submission" / name)
    for name in module.MANUSCRIPT_FILES: _write(tmp_path / "manuscript" / name)
    for name in module.TEMPLATE_FILES: _write(tmp_path / "templates" / name)
    _write(tmp_path / "templates/debug_template.csv")
    _write(tmp_path / "src/pkg/a.py")
    _write(tmp_path / "src/pkg/__pycache__/a.cpython-313.pyc")
    for name in module.SCRIPT_FILES: _write(tmp_path / "scripts" / name)
    for name in module.TEST_FILES: _write(tmp_path / "tests" / name)
    _write(tmp_path / "scripts/debug_experiment.py")
    _write(tmp_path / "scripts/__pycache__/debug_experiment.cpython-313.pyc")
    _write(tmp_path / "tests/test_debug_experiment.py")
    _write(tmp_path / "manuscript/elsarticle-template-num.aux")
    _write(tmp_path / "generated_submission/runtime_cap_rerun_work/x.csv")
    _write(tmp_path / "generated_submission/friedman_summary.csv")
    paths = {p.relative_to(tmp_path).as_posix() for p in module.collect_supplement_paths(tmp_path)}
    assert "src/pkg/a.py" in paths
    assert ".gitignore" in paths
    assert all(f"scripts/{name}" in paths for name in module.SCRIPT_FILES)
    assert all(f"tests/{name}" in paths for name in module.TEST_FILES)
    assert all(f"templates/{name}" in paths for name in module.TEMPLATE_FILES)
    assert "templates/debug_template.csv" not in paths
    assert "scripts/debug_experiment.py" not in paths
    assert "tests/test_debug_experiment.py" not in paths
    assert not any("__pycache__" in p for p in paths)
    assert "manuscript/elsarticle-template-num.aux" not in paths
    assert "generated_submission/runtime_cap_rerun_work/x.csv" not in paths
    assert "generated_submission/friedman_summary.csv" not in paths


def test_package_status_has_no_wall_clock_timestamp(tmp_path: Path) -> None:
    _write(tmp_path / "generated_submission/evidence_provenance.json", json.dumps({"schema":"x"}))
    _write(tmp_path / "manuscript/build_provenance.json", json.dumps({"schema":"x"}))
    _write(tmp_path / "manuscript/graphical_abstract_provenance.json", json.dumps({"schema":"ga"}))
    status = module._package_status(tmp_path).decode()
    assert "status=CLOSED" in status
    assert "graphical_abstract_provenance_sha256=" in status
    assert "generated_utc" not in status and "built_utc" not in status


def test_portal_contract_records_stable_copy_policy() -> None:
    text = module._contract_md("portal").decode("utf-8")
    assert "85 characters" in text
    assert "rerun-sensitive percentages/test statistics" in text
    assert "regenerated evidence" in text


def test_graphical_abstract_provenance_is_part_of_release_contract() -> None:
    assert "graphical_abstract_provenance.json" in module.MANUSCRIPT_FILES
    assert "compile_graphical_abstract_release.py" in module.SCRIPT_FILES


def test_preflight_verifies_graphical_abstract_provenance(monkeypatch, tmp_path: Path) -> None:
    calls = []
    def fake_run(args, cwd, check):
        calls.append(tuple(args))
        class Result: pass
        return Result()
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "require_closed_state", lambda root: None)
    module.run_preflight(tmp_path)
    assert any("compile_manuscript_release.py" in call for args in calls for call in args)
    assert any("compile_graphical_abstract_release.py" in call for args in calls for call in args)


def test_strengthening_evidence_is_part_of_release_contract() -> None:
    assert "build_submission_strengthening_evidence.py" in module.SCRIPT_FILES
    required = {
        "protocol_sensitivity.csv",
        "runtime_cap_deviations.csv",
        "block_effect_bootstrap_ci.csv",
        "stability_by_method.csv",
        "historical_variant_directional_diagnostics.csv",
        "strengthening_evidence_provenance.json",
    }
    assert required.issubset(set(module.GENERATED_FILES))


def test_operational_shift_evidence_is_part_of_release_contract() -> None:
    assert "build_operational_shift_evidence.py" in module.SCRIPT_FILES
    assert "operational_shift_stress_locked.csv" in module.CONFIG_FILES
    assert "solomon_equal_3shift_feasibility_audit.csv" in module.CONFIG_FILES
    required={"operational_shift_stress_summary.csv","table_operational_shift_stress.tex","operational_shift_evidence_provenance.json"}
    assert required.issubset(set(module.GENERATED_FILES))


def test_parameter_sensitivity_evidence_is_part_of_release_contract() -> None:
    assert "build_parameter_sensitivity_evidence.py" in module.SCRIPT_FILES
    assert "parameter_sensitivity_locked.csv" in module.CONFIG_FILES
    required={"parameter_sensitivity_summary.csv","parameter_sensitivity_stats.csv","table_parameter_sensitivity.tex","parameter_sensitivity_evidence_provenance.json"}
    assert required.issubset(set(module.GENERATED_FILES))


def test_anytime_profile_evidence_is_part_of_release_contract() -> None:
    assert "build_anytime_profile_evidence.py" in module.SCRIPT_FILES
    assert "anytime_profile_locked.csv" in module.CONFIG_FILES
    required = {
        "anytime_profile_summary.csv",
        "anytime_profile_auc.csv",
        "anytime_profile_overall.csv",
        "table_anytime_profile.tex",
        "anytime_profile_evidence_provenance.json",
    }
    assert required.issubset(set(module.GENERATED_FILES))
