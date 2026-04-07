#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pathlib import Path
import csv
from swevo_suite.paths import GENERATED

state_path = GENERATED / "run_state.csv"
if not state_path.exists():
    raise SystemExit("generated/run_state.csv not found; run scripts/init_run_state.py first.")

pending = []
with state_path.open() as f:
    for row in csv.DictReader(f):
        if row["status"] in {"pending", "failed"}:
            pending.append(row["run_id"])
print(f"Pending or failed runs: {len(pending)}")
for run_id in pending[:50]:
    print(run_id)
print("Use run_id list to dispatch batch reruns in the real cluster/job scheduler.")