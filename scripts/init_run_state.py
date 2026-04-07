#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import csv
from swevo_suite.manifest import load_manifest
from swevo_suite.checkpoint import init_run_state

if __name__ == "__main__":
    rows = [r.__dict__ for r in load_manifest()]
    out = init_run_state(rows)
    print(f"Wrote {out}")