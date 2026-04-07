#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from pathlib import Path
import pandas as pd
from swevo_suite.paths import GENERATED, ensure_generated_dirs
from swevo_suite.latex import dataframe_to_latex_table

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", nargs="?", default=str(GENERATED / "summary_by_method.csv"))
    parser.add_argument("stats", nargs="?", default=str(GENERATED / "master_stats.csv"))
    parser.add_argument("--output-dir", default=str(GENERATED))
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    ensure_generated_dirs()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(Path(args.summary))
    stats = pd.read_csv(Path(args.stats))

    feas = summary[["tier","method_id","runs","accepted_rate","strict_duty_rate","median_j","median_wh"]]
    feas_tex = dataframe_to_latex_table(
        feas, "Feasibility and efficiency summary generated from master_runs.csv", "tab:feasibility_summary_generated"
    )
    (out_dir / f"table_feasibility_summary{args.suffix}.tex").write_text(feas_tex)

    pairwise = stats[stats["section"] == "pairwise"][["compare_method","n_pairs","p_value","p_holm","median_diff_control_minus_compare","rank_biserial"]]
    pairwise_tex = dataframe_to_latex_table(
        pairwise, "Pairwise control-vs-others results for EDE", "tab:posthoc_holm_generated"
    )
    (out_dir / f"table_posthoc_holm{args.suffix}.tex").write_text(pairwise_tex)

    omnibus = stats[stats["section"] == "omnibus"][["metric","blocks","friedman_chi2","p_value","kendalls_w"]]
    omnibus_tex = dataframe_to_latex_table(
        omnibus, "Across-instance omnibus test results", "tab:friedman_omnibus_generated"
    )
    (out_dir / f"table_friedman_omnibus{args.suffix}.tex").write_text(omnibus_tex)

    print("Wrote generated LaTeX tables")


if __name__ == "__main__":
    main()
