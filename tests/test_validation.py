from pathlib import Path
import csv
import subprocess
import sys

def test_template_has_header():
    path = Path(__file__).resolve().parents[1] / "templates" / "master_runs_template.csv"
    header = path.read_text().strip().split(",")
    assert "accepted_final" in header
    assert "strict_duty_final" in header


def test_generated_dir_creation_does_not_recreate_legacy_subdirs(tmp_path, monkeypatch):
    from importlib.util import module_from_spec, spec_from_file_location

    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "generated_submission"
    monkeypatch.setenv("SWEVO_GENERATED_DIR", str(target))
    spec = spec_from_file_location("swevo_paths_test", root / "src" / "swevo_suite" / "paths.py")
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module.ensure_generated_dirs()
    assert target.is_dir()
    for name in ("checkpoints", "archives", "figures"):
        assert not (target / name).exists()


def _load_validator():
    from importlib.util import module_from_spec, spec_from_file_location
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location("validate_master_runs_test", root / "scripts" / "validate_master_runs.py")
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_runtime_fixture(tmp_path: Path, run_ids: list[str]) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    required = [
        "run_id","accepted_final","strict_duty_final","v_cap_final","v_tw_final","v_shift_final",
        "overtime_sum_final","j_scaled_init","j_scaled_final","compute_wh","imp_per_wh","runtime_sec",
    ]
    ledger = tmp_path / "master.csv"
    manifest = tmp_path / "manifest.csv"
    with ledger.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=required)
        writer.writeheader()
        for rid in run_ids:
            writer.writerow({
                "run_id": rid,
                "accepted_final": 1,
                "strict_duty_final": 1,
                "v_cap_final": 0,
                "v_tw_final": 0,
                "v_shift_final": 0,
                "overtime_sum_final": 0,
                "j_scaled_init": 2,
                "j_scaled_final": 1,
                "compute_wh": 1,
                "imp_per_wh": 1,
                "runtime_sec": 8000,
            })
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "walltime_cap_s"])
        writer.writeheader()
        for rid in run_ids:
            writer.writerow({"run_id": rid, "walltime_cap_s": 7200})
    return ledger, manifest


def test_validator_strict_mode_rejects_locked_runtime_exception(tmp_path: Path) -> None:
    module = _load_validator()
    ids = sorted(module.LOCKED_RUNTIME_EXCEPTION_IDS)
    ledger, manifest = _write_runtime_fixture(tmp_path, ids)
    assert module.main(str(ledger), str(manifest), 10.0, False) == 1


def test_validator_allows_only_exact_locked_runtime_exception_when_explicit(tmp_path: Path) -> None:
    module = _load_validator()
    ids = sorted(module.LOCKED_RUNTIME_EXCEPTION_IDS)
    ledger, manifest = _write_runtime_fixture(tmp_path, ids)
    assert module.main(str(ledger), str(manifest), 10.0, True) == 0

    ledger2, manifest2 = _write_runtime_fixture(tmp_path / "extra", ids + ["unexpected"])
    assert module.main(str(ledger2), str(manifest2), 10.0, True) == 1
