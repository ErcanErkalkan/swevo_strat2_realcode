from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "propose_benchmark_repairs.py"


def _load_script_module():
    spec = spec_from_file_location("propose_benchmark_repairs", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_repair_plan_prefers_unused_same_series_candidate() -> None:
    module = _load_script_module()

    inventory_rows = [
        {"benchmark_family": "solomon", "customer_count": "100", "instance_id": "RC101", "structure_class": "mixed", "default_tier": "small"},
        {"benchmark_family": "solomon", "customer_count": "100", "instance_id": "RC104", "structure_class": "mixed", "default_tier": "small"},
        {"benchmark_family": "solomon", "customer_count": "100", "instance_id": "RC109", "structure_class": "mixed", "default_tier": "small"},
        {"benchmark_family": "solomon", "customer_count": "100", "instance_id": "RC201", "structure_class": "mixed", "default_tier": "small"},
    ]
    resolution_rows = [
        {
            "instance_id": "RC109",
            "status": "invalid_public_instance_id",
            "error": "InvalidBenchmarkInstanceId: RC109 is not part of the public Solomon-100 benchmark set.",
        }
    ]
    manifest_rows = [
        {"instance_id": "RC109"},
        {"instance_id": "RC109"},
        {"instance_id": "C101"},
    ]

    suggestions, replacement_map = module.build_repair_plan(inventory_rows, resolution_rows, manifest_rows)

    assert len(suggestions) == 1
    assert suggestions[0]["suggestion_1"] == "RC108"
    assert suggestions[0]["blocked_manifest_rows"] == 2
    assert replacement_map == {"RC109": "RC108"}
