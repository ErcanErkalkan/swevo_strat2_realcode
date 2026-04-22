from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_benchmark_inventory.py"


def _load_script_module():
    spec = spec_from_file_location("check_benchmark_inventory", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inventory_resolution_flags_invalid_public_solomon_id() -> None:
    module = _load_script_module()

    valid = module._resolve_inventory_row(
        {
            "instance_id": "RC108",
            "benchmark_family": "solomon",
            "customer_count": "100",
            "default_tier": "small",
        }
    )
    invalid = module._resolve_inventory_row(
        {
            "instance_id": "RC109",
            "benchmark_family": "solomon",
            "customer_count": "100",
            "default_tier": "small",
        }
    )

    assert valid["status"] == "ok"
    assert invalid["status"] == "invalid_public_instance_id"
    assert "RC101-RC108" in invalid["error"]
