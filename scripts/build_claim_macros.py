#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pathlib import Path
import pandas as pd
from swevo_suite.paths import GENERATED, ensure_generated_dirs

def fmt(x: float) -> str:
    return f"{x:.2f}"

if __name__ == "__main__":
    ensure_generated_dirs()
    df = pd.read_csv(GENERATED / "summary_by_method.csv")
    lines = []
    for tier in sorted(df["tier"].unique()):
        row = df[(df["method_id"] == "EDE") & (df["tier"] == tier)]
        if row.empty:
            continue
        r = row.iloc[0]
        suffix = tier.capitalize()
        lines.append(f"\\newcommand{{\\AccRateEDE{suffix}}}{{{fmt(r['accepted_rate'])}}}")
        lines.append(f"\\newcommand{{\\StrictDutyEDE{suffix}}}{{{fmt(r['strict_duty_rate'])}}}")
        lines.append(f"\\newcommand{{\\MedianJEDE{suffix}}}{{{fmt(r['median_j'])}}}")
        lines.append(f"\\newcommand{{\\MedianWhEDE{suffix}}}{{{fmt(r['median_wh'])}}}")
    out = GENERATED / "claim_macros.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")