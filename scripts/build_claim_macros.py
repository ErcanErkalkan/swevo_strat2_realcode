#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
import pandas as pd
from swevo_suite.paths import GENERATED, ensure_generated_dirs

def fmt(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.2f}"


def tex_token(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isalnum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", nargs="?", default=str(GENERATED / "summary_by_method.csv"))
    parser.add_argument("--accepted-summary", default=None)
    parser.add_argument("--output", default=str(GENERATED / "claim_macros.tex"))
    args = parser.parse_args()

    ensure_generated_dirs()
    summary_path = Path(args.summary)
    output_path = Path(args.output)
    accepted_path = (
        Path(args.accepted_summary)
        if args.accepted_summary
        else summary_path.with_name(f"{summary_path.stem}_accepted_only{summary_path.suffix}")
    )
    overall = pd.read_csv(summary_path)
    accepted = pd.read_csv(accepted_path) if accepted_path.exists() else overall
    lines = []
    keys = sorted({(row.method_id, row.tier) for row in overall.itertuples(index=False)})
    for method_id, tier in keys:
        token = tex_token(method_id)
        suffix = tex_token(tier.capitalize())
        overall_row = overall[(overall["method_id"] == method_id) & (overall["tier"] == tier)]
        accepted_row = accepted[(accepted["method_id"] == method_id) & (accepted["tier"] == tier)]
        if overall_row.empty:
            continue
        overall_item = overall_row.iloc[0]
        accepted_item = accepted_row.iloc[0] if not accepted_row.empty else overall_item
        lines.append(f"\\newcommand{{\\AccRate{token}{suffix}}}{{{fmt(overall_item['accepted_rate'])}}}")
        lines.append(f"\\newcommand{{\\StrictDuty{token}{suffix}}}{{{fmt(overall_item['strict_duty_rate'])}}}")
        lines.append(f"\\newcommand{{\\MedianJ{token}{suffix}}}{{{fmt(accepted_item['median_j'])}}}")
        lines.append(f"\\newcommand{{\\MedianWh{token}{suffix}}}{{{fmt(accepted_item['median_wh'])}}}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
