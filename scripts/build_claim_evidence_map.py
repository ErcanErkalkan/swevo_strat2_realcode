#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import pandas as pd

from swevo_suite.paths import GENERATED, TEMPLATES, ensure_generated_dirs


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _default_accepted_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_accepted_only{path.suffix}")


def _summary_row(summary: pd.DataFrame, method_id: str) -> pd.Series | None:
    rows = summary[summary["method_id"] == method_id]
    if rows.empty:
        return None
    return rows.sort_values(["phase", "tier"]).iloc[0]


def _pairwise_row(stats: pd.DataFrame, compare_method: str) -> pd.Series | None:
    rows = stats[(stats["section"] == "pairwise") & (stats["compare_method"] == compare_method)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _omnibus_row(stats: pd.DataFrame) -> pd.Series | None:
    rows = stats[stats["section"] == "omnibus"]
    if rows.empty:
        return None
    return rows.iloc[0]


def _first_non_null(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _fill_claim_row(row: pd.Series, summary: pd.DataFrame | None, stats: pd.DataFrame | None) -> pd.Series:
    claim_id = str(row["claim_id"]).strip().upper()
    out = row.copy()
    out["status"] = "missing_artifacts"
    out["notes"] = "Required artifacts were not available."

    if claim_id == "C01" and summary is not None:
        ede = _summary_row(summary, "EDE")
        if ede is not None:
            out["status"] = "supported_by_artifacts"
            out["notes"] = (
                f"EDE accepted_rate={_fmt_pct(ede['accepted_rate'])}; "
                f"strict_duty_rate={_fmt_pct(ede['strict_duty_rate'])}; "
                f"scope={ede.get('summary_scope', 'unconditional')}"
            )
        return out

    if claim_id == "C02" and stats is not None:
        baseline = _first_non_null(_pairwise_row(stats, "StdDE"), _pairwise_row(stats, "ILS_MS"))
        if baseline is not None:
            supported = pd.notna(baseline["p_holm"]) and baseline["p_holm"] <= 0.05 and baseline["median_diff_control_minus_compare"] < 0
            out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
            out["notes"] = (
                f"EDE vs {baseline['compare_method']}: "
                f"median_diff={_fmt_float(baseline['median_diff_control_minus_compare'])}; "
                f"p_holm={_fmt_float(baseline['p_holm'])}; "
                f"scope={baseline.get('summary_scope', 'unconditional')}"
            )
        return out

    if claim_id == "C03" and stats is not None:
        omnibus = _omnibus_row(stats)
        if omnibus is not None:
            supported = pd.notna(omnibus["p_value"]) and omnibus["p_value"] <= 0.05
            out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
            out["notes"] = (
                f"Friedman blocks={int(omnibus['blocks']) if pd.notna(omnibus['blocks']) else 0}; "
                f"chi2={_fmt_float(omnibus['friedman_chi2']) if pd.notna(omnibus['friedman_chi2']) else 'NA'}; "
                f"p={_fmt_float(omnibus['p_value']) if pd.notna(omnibus['p_value']) else 'NA'}; "
                f"scope={omnibus.get('summary_scope', 'unconditional')}"
            )
        return out

    if claim_id == "C04" and summary is not None:
        ede = _summary_row(summary, "EDE")
        stdde = _summary_row(summary, "StdDE")
        if ede is not None and stdde is not None:
            supported = pd.notna(ede["median_imp_per_wh"]) and pd.notna(stdde["median_imp_per_wh"]) and ede["median_imp_per_wh"] > stdde["median_imp_per_wh"]
            out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
            out["notes"] = (
                f"EDE median_imp_per_wh={_fmt_float(ede['median_imp_per_wh'])}; "
                f"StdDE median_imp_per_wh={_fmt_float(stdde['median_imp_per_wh'])}; "
                f"scope={ede.get('summary_scope', 'unconditional')}"
            )
        return out

    out["status"] = "manual_review_needed"
    out["notes"] = "Claim id not yet wired into the generator."
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", nargs="?", default=str(TEMPLATES / "claim_evidence_map_template.csv"))
    parser.add_argument("--summary", default=str(GENERATED / "summary_by_method.csv"))
    parser.add_argument("--stats", default=str(GENERATED / "master_stats.csv"))
    parser.add_argument("--accepted-summary", default=None)
    parser.add_argument("--accepted-stats", default=None)
    parser.add_argument("--output", default=str(GENERATED / "claim_evidence_map.csv"))
    args = parser.parse_args()

    ensure_generated_dirs()
    template_path = Path(args.template)
    summary_path = Path(args.summary)
    stats_path = Path(args.stats)
    accepted_summary_path = Path(args.accepted_summary) if args.accepted_summary else _default_accepted_path(summary_path)
    accepted_stats_path = Path(args.accepted_stats) if args.accepted_stats else _default_accepted_path(stats_path)
    output_path = Path(args.output)

    template = pd.read_csv(template_path)
    summary = _first_non_null(_load_optional_csv(accepted_summary_path), _load_optional_csv(summary_path))
    stats = _first_non_null(_load_optional_csv(accepted_stats_path), _load_optional_csv(stats_path))

    completed = template.apply(_fill_claim_row, axis=1, summary=summary, stats=stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
