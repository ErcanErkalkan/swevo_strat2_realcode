from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT = ROOT / "scripts" / "merge_runtime_cap_rerun.py"
LAUNCHER_SCRIPT = ROOT / "scripts" / "run_runtime_cap_rerun_parallel.py"
EXPECTED_IDS = [
    "R1_4_5__S1_balanced__EDE__seed20",
    "R1_4_5__S1_balanced__StdDE__seed20",
    "R1_4_5__S1_balanced__ALNS_MS__seed20",
    "R1_4_5__S1_balanced__HGS_MS__seed20",
    "R1_4_5__S1_balanced__ILS_MS__seed20",
]


def _load_launcher_module():
    spec = spec_from_file_location("run_runtime_cap_rerun_parallel", LAUNCHER_SCRIPT)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module




def test_launcher_defaults_keep_rerun_artifacts_inside_remediation_surface() -> None:
    module = _load_launcher_module()
    assert module.DEFAULT_OUTPUT == (
        "EDE_RUNTIME_REMEDIATION_2026-09-03/"
        "runtime_cap_rerun_R1_4_5_S1_seed20_results.csv"
    )
    assert module.DEFAULT_WORK_DIR == "EDE_RUNTIME_REMEDIATION_2026-09-03/work"
    assert "generated_submission" not in module.DEFAULT_WORK_DIR


def _rerun_frame(runtime=7000.0):
    return pd.DataFrame([
        {
            "run_id": run_id,
            "runtime_sec": runtime,
            "accepted_final": 1,
            "strict_duty_final": 1,
            "objective_j": float(i),
        }
        for i, run_id in enumerate(EXPECTED_IDS)
    ])


def test_locked_manifest_contract_rejects_wrong_method_set(tmp_path: Path) -> None:
    module = _load_launcher_module()
    manifest = tmp_path / "rerun.csv"
    rows = []
    for run_id, method in zip(EXPECTED_IDS, ["EDE", "StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"]):
        rows.append({
            "run_id": run_id,
            "instance_id": "R1_4_5",
            "scenario_id": "S1_balanced",
            "method_id": method,
            "seed": 20,
            "walltime_cap_s": 7200,
        })
    rows[-1]["method_id"] = "WRONG"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    try:
        module._load_locked_manifest(manifest)
    except SystemExit:
        pass
    else:
        raise AssertionError("Locked rerun manifest accepted a wrong method set")


def test_merge_rejects_nan_runtime_and_does_not_write_output(tmp_path: Path) -> None:
    master = pd.concat([
        _rerun_frame(),
        pd.DataFrame([{
            "run_id": "OTHER",
            "runtime_sec": 1.0,
            "accepted_final": 1,
            "strict_duty_final": 1,
            "objective_j": 99.0,
        }]),
    ], ignore_index=True)
    rerun = _rerun_frame()
    rerun.loc[0, "runtime_sec"] = float("nan")
    master_path = tmp_path / "master.csv"
    rerun_path = tmp_path / "rerun.csv"
    output_path = tmp_path / "out.csv"
    master.to_csv(master_path, index=False)
    rerun.to_csv(rerun_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(MERGE_SCRIPT),
            "--master", str(master_path),
            "--rerun", str(rerun_path),
            "--output", str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "missing/non-finite/non-positive" in (proc.stdout + proc.stderr)
    assert not output_path.exists()


def test_merge_preserves_cardinality_ids_and_order(tmp_path: Path) -> None:
    master = pd.concat([
        _rerun_frame(runtime=7100.0),
        pd.DataFrame([{
            "run_id": "OTHER",
            "runtime_sec": 1.0,
            "accepted_final": 1,
            "strict_duty_final": 1,
            "objective_j": 99.0,
        }]),
    ], ignore_index=True)
    rerun = _rerun_frame(runtime=7000.0)
    master_path = tmp_path / "master.csv"
    rerun_path = tmp_path / "rerun.csv"
    output_path = tmp_path / "out.csv"
    master.to_csv(master_path, index=False)
    rerun.to_csv(rerun_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(MERGE_SCRIPT),
            "--master", str(master_path),
            "--rerun", str(rerun_path),
            "--output", str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = pd.read_csv(output_path)
    assert out["run_id"].tolist() == master["run_id"].tolist()
    assert out["run_id"].is_unique
    assert len(out) == len(master)
    assert not list(tmp_path.glob(".out.csv.*.tmp"))
