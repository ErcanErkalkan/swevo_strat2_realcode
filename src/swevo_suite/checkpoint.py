from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable
from .paths import GENERATED

STATE_COLUMNS = [
    "run_id", "phase", "instance_id", "scenario_id", "method_id", "seed",
    "status", "attempt_count", "last_update_utc", "last_error", "output_row_written"
]

def init_run_state(run_ids: Iterable[dict], path: Path | None = None) -> Path:
    out = path or (GENERATED / "run_state.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATE_COLUMNS)
        writer.writeheader()
        for row in run_ids:
            writer.writerow({
                "run_id": row["run_id"],
                "phase": row["phase"],
                "instance_id": row["instance_id"],
                "scenario_id": row["scenario_id"],
                "method_id": row["method_id"],
                "seed": row["seed"],
                "status": "pending",
                "attempt_count": 0,
                "last_update_utc": "",
                "last_error": "",
                "output_row_written": 0,
            })
    return out
