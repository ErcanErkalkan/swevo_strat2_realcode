#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from swevo_suite.manifest import load_manifest, write_run_matrix

if __name__ == "__main__":
    rows = load_manifest()
    out = write_run_matrix(rows)
    print(f"Wrote {out}")