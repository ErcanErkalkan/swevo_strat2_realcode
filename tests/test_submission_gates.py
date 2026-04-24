from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_submission_gates.py"


def _load_script_module():
    spec = spec_from_file_location("check_submission_gates", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gate_helpers_return_expected_status_labels() -> None:
    module = _load_script_module()
    assert module._pass(1, "x", "y")["status"] == "pass"
    assert module._fail(1, "x", "y")["status"] == "fail"
    assert module._manual(1, "x", "y")["status"] == "manual_review_needed"


def test_write_markdown_renders_status_table(tmp_path: Path) -> None:
    module = _load_script_module()
    rows = [
        {"item_id": 1, "title": "Example gate", "status": "pass", "notes": "ok"},
        {"item_id": 2, "title": "Another gate", "status": "manual_review_needed", "notes": "manual"},
    ]
    out = tmp_path / "report.md"
    module.write_markdown(out, rows)
    text = out.read_text(encoding="utf-8")
    assert "Submission Gating Report" in text
    assert "| 1. Example gate | pass | ok |" in text
