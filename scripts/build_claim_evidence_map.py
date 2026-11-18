#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import pandas as pd

from swevo_suite.paths import GENERATED, TEMPLATES, ensure_generated_dirs


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _fmt_p(value: float) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3e}"


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _default_accepted_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_accepted_only{path.suffix}")


def _pairwise_row(stats: pd.DataFrame, compare_method: str) -> pd.Series | None:
    rows = stats[(stats["section"] == "pairwise") & (stats["compare_method"] == compare_method)]
    return None if rows.empty else rows.iloc[0]


def _omnibus_row(stats: pd.DataFrame) -> pd.Series | None:
    rows = stats[stats["section"] == "omnibus"]
    return None if rows.empty else rows.iloc[0]


def _first_non_null(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _fill_claim_row(
    row: pd.Series,
    summary: pd.DataFrame | None,
    stats: pd.DataFrame | None,
    scenario: pd.DataFrame | None = None,
    structure: pd.DataFrame | None = None,
    native: pd.DataFrame | None = None,
    protocol: pd.DataFrame | None = None,
    ci: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
    historical: pd.DataFrame | None = None,
    operational: pd.DataFrame | None = None,
    parameter_sensitivity: pd.DataFrame | None = None,
    parameter_sensitivity_stats: pd.DataFrame | None = None,
    anytime_summary: pd.DataFrame | None = None,
    anytime_auc: pd.DataFrame | None = None,
    anytime_overall: pd.DataFrame | None = None,
) -> pd.Series:
    claim_id = str(row["claim_id"]).strip().upper()
    out = row.copy()
    out["status"] = "missing_artifacts"
    out["notes"] = "Required submission-primary artifacts were not available."

    if claim_id == "C01" and summary is not None:
        primary = {"EDE", "StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"}
        present = set(summary["method_id"].dropna().astype(str))
        scoped = summary[summary["method_id"].isin(primary)].copy()
        methods_ok = present == primary and set(scoped["method_id"].astype(str)) == primary
        accepted_ok = (scoped["accepted_rate"] == 100.0).all() if not scoped.empty else False
        strict_ok = (scoped["strict_duty_rate"] == 100.0).all() if not scoped.empty else False
        if "source_runs" in scoped:
            run_totals = scoped.groupby("method_id")["source_runs"].sum().astype(int).to_dict()
            runs_ok = all(run_totals.get(m) == 2160 for m in primary)
        else:
            run_totals = {}
            runs_ok = False
        supported = methods_ok and accepted_ok and strict_ok and runs_ok
        out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
        out["notes"] = (
            f"primary_methods={','.join(sorted(present))}; methods_exact={methods_ok}; "
            f"accepted_all_100={accepted_ok}; strict_duty_all_100={strict_ok}; "
            f"source_runs_by_method={run_totals}."
        )
        return out

    if claim_id == "C02" and stats is not None:
        comparisons = ["StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"]
        rows = [_pairwise_row(stats, method) for method in comparisons]
        rows = [r for r in rows if r is not None]
        supported = (
            len(rows) == 4
            and all(pd.notna(r["p_holm"]) and r["p_holm"] <= 0.05 for r in rows)
            and all(r["median_diff_control_minus_compare"] < 0 for r in rows)
            and all(int(r["n_pairs"]) == 108 for r in rows)
        )
        out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
        out["notes"] = "; ".join(
            f"{r['compare_method']}: diff={_fmt_float(r['median_diff_control_minus_compare'])}, "
            f"p_holm={_fmt_p(r['p_holm'])}, n={int(r['n_pairs'])}" for r in rows
        ) if rows else "Primary pairwise rows missing."
        return out

    if claim_id == "C03" and stats is not None:
        omnibus = _omnibus_row(stats)
        if omnibus is not None:
            supported = (
                pd.notna(omnibus["p_value"]) and omnibus["p_value"] <= 0.05
                and int(omnibus["blocks"]) == 108
                and int(omnibus.get("k_methods", 0)) == 5
            )
            out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
            out["notes"] = (
                f"blocks={int(omnibus['blocks'])}; k={int(omnibus.get('k_methods', 0))}; "
                f"chi2={_fmt_float(omnibus['friedman_chi2'])}; "
                f"p={_fmt_p(omnibus['p_value'])}; "
                f"Kendall_W={_fmt_float(omnibus['kendalls_w'])}"
            )
        return out

    if claim_id == "C04" and summary is not None:
        has_proxy = "median_wh" in summary.columns and summary["median_wh"].notna().all()
        out["status"] = "supported_by_code_and_artifacts" if has_proxy else "missing_artifacts"
        out["notes"] = (
            "compute_wh is reported as a runtime-based model proxy; the manuscript explicitly disclaims "
            "hardware-metered electrical-energy interpretation."
            if has_proxy else "median_wh proxy artifact is missing."
        )
        return out

    if claim_id == "C05" and scenario is not None:
        supported = (
            len(scenario) == 3
            and (scenario["blocks"] == 36).all()
            and (scenario["ede_reduction_vs_stdde_pct"] > 0).all()
            and (scenario["ede_reduction_vs_best_other_pct"] > 0).all()
        )
        out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
        if not scenario.empty:
            out["notes"] = (
                f"scenarios={len(scenario)}; blocks_each={sorted(scenario['blocks'].unique().tolist())}; "
                f"EDE_vs_StdDE_reduction_range={scenario['ede_reduction_vs_stdde_pct'].min():.2f}%--"
                f"{scenario['ede_reduction_vs_stdde_pct'].max():.2f}%; "
                f"EDE_vs_best_other_range={scenario['ede_reduction_vs_best_other_pct'].min():.2f}%--"
                f"{scenario['ede_reduction_vs_best_other_pct'].max():.2f}%."
            )
        return out

    if claim_id == "C06" and structure is not None:
        expected = {"clustered", "mixed", "random"}
        supported = (
            set(structure["structure_class"].astype(str)) == expected
            and (structure["blocks"] == 36).all()
            and (structure["ede_reduction_vs_stdde_pct"] > 0).all()
            and (structure["ede_reduction_vs_best_other_pct"] > 0).all()
        )
        out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
        if not structure.empty:
            out["notes"] = (
                f"classes={','.join(sorted(structure['structure_class'].astype(str)))}; "
                f"EDE_vs_StdDE_reduction_range={structure['ede_reduction_vs_stdde_pct'].min():.2f}%--"
                f"{structure['ede_reduction_vs_stdde_pct'].max():.2f}%; "
                f"EDE_vs_best_other_range={structure['ede_reduction_vs_best_other_pct'].min():.2f}%--"
                f"{structure['ede_reduction_vs_best_other_pct'].max():.2f}%."
            )
        return out

    if claim_id == "C07" and native is not None:
        target_metrics = {"j_scaled_final", "cost_final", "energy_final", "co2_final"}
        subset = native[native["metric"].isin(target_metrics)].copy()
        supported = (
            len(subset) == 12
            and set(subset["tier"].astype(str)) == {"small", "medium", "large"}
            and set(subset["metric"].astype(str)) == target_metrics
            and (subset["reduction_vs_stdde_pct"] > 0).all()
        )
        out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
        if not subset.empty:
            out["notes"] = (
                f"tier_metric_rows={len(subset)}; all_reductions_positive={bool((subset['reduction_vs_stdde_pct'] > 0).all())}; "
                f"range={subset['reduction_vs_stdde_pct'].min():.2f}%--{subset['reduction_vs_stdde_pct'].max():.2f}%."
            )
        return out


    if claim_id == "C08" and protocol is not None:
        required_scopes = {"all_10800", "cap_compliant_10795"}
        scopes_ok = set(protocol["scope"].astype(str)) == required_scopes
        raw = protocol[protocol["scope"] == "all_10800"].sort_values(["section", "compare_method"], na_position="first").reset_index(drop=True)
        cmp = protocol[protocol["scope"] == "cap_compliant_10795"].sort_values(["section", "compare_method"], na_position="first").reset_index(drop=True)
        numeric_cols = ["statistic", "p_value", "kendalls_w", "median_diff_ede_minus_compare", "p_holm"]
        supported = scopes_ok and len(raw) == 5 and len(cmp) == 5
        if supported:
            for col in numeric_cols:
                a = pd.to_numeric(raw[col], errors="coerce").to_numpy()
                b = pd.to_numeric(cmp[col], errors="coerce").to_numpy()
                if not __import__("numpy").allclose(a, b, equal_nan=True, rtol=0.0, atol=1e-15):
                    supported = False
                    break
        out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
        out["notes"] = (
            "raw_10800 and paired cap-compliant_10795 inference rows match exactly for omnibus and four pairwise comparisons."
            if supported else "Protocol sensitivity artifact is missing expected scopes or inferential equality."
        )
        return out

    if claim_id == "C09" and ci is not None:
        supported = (
            set(ci["compare_method"].astype(str)) == {"StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"}
            and (pd.to_numeric(ci["blocks"], errors="coerce") == 108).all()
            and (pd.to_numeric(ci["bootstrap_ci95_high"], errors="coerce") < 0).all()
        )
        out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
        if supported:
            out["notes"] = "; ".join(
                f"{r.compare_method}: median={float(r.median_diff_ede_minus_compare):.6f}, "
                f"CI95=[{float(r.bootstrap_ci95_low):.6f},{float(r.bootstrap_ci95_high):.6f}], "
                f"EDE_better_blocks={float(r.ede_better_blocks_pct):.1f}%"
                for r in ci.itertuples(index=False)
            )
        return out

    if claim_id == "C10" and stability is not None:
        rows = stability.set_index("method_id") if "method_id" in stability.columns else pd.DataFrame()
        supported = (
            not rows.empty and "EDE" in rows.index
            and set(["EDE", "StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"]).issubset(set(rows.index.astype(str)))
        )
        if supported:
            ede_iqr = float(rows.loc["EDE", "median_within_block_iqr"])
            min_iqr = float(pd.to_numeric(rows["median_within_block_iqr"], errors="coerce").min())
            supported = abs(ede_iqr - min_iqr) <= 1e-12
        out["status"] = "supported_by_artifacts" if supported else "not_supported_by_artifacts"
        if supported:
            out["notes"] = (
                f"EDE median within-block IQR={ede_iqr:.6f}; "
                f"five-method minimum={min_iqr:.6f}."
            )
        return out

    if claim_id == "C11" and historical is not None:
        supported = (
            set(historical["variant"].astype(str)) == {"A1_NoSeed", "A2_NoJDE", "A3_NoLNS"}
            and (pd.to_numeric(historical["blocks"], errors="coerce") == 108).all()
            and (pd.to_numeric(historical["ede_better_blocks_pct"], errors="coerce") == 100.0).all()
            and historical["interpretation"].astype(str).str.contains("not_causal", case=False).all()
        )
        out["status"] = "supported_as_exploratory_noncausal" if supported else "not_supported_by_artifacts"
        out["notes"] = (
            "All three historical variants are directionally worse in 108/108 blocks; artifact explicitly labels the comparison exploratory and non-causal."
            if supported else "Historical directional artifact is missing expected scope/non-causal labeling."
        )
        return out

    if claim_id == "C12" and operational is not None:
        rows = operational[operational["scope"].astype(str) == "all"]
        supported = False
        notes = "Operational shift summary is missing the all-scope row."
        if len(rows) == 1:
            r = rows.iloc[0]
            supported = (
                int(r["runs"]) == 18
                and int(r["hard_feasible_runs"]) == 18
                and int(r["strict_duty_runs"]) == 15
                and int(r["bounded_overtime_runs"]) == 3
                and float(r["max_v_cap"]) <= 1e-9
                and float(r["max_v_tw"]) <= 1e-9
                and float(r["max_v_shift"]) <= 1e-9
                and float(r["max_overtime_sum"]) <= 0.35
            )
            notes = (
                f"secondary non-overlapping-shift stress: hard_feasible={int(r['hard_feasible_runs'])}/{int(r['runs'])}; "
                f"strict_zero_overtime={int(r['strict_duty_runs'])}/{int(r['runs'])}; "
                f"bounded_overtime_runs={int(r['bounded_overtime_runs'])}; max_overtime={float(r['max_overtime_sum']):.6f}; "
                "scope is implementation-level operational semantics, not real-world deployment or optimizer superiority."
            )
        out["status"] = "supported_as_secondary_semantics" if supported else "not_supported_by_artifacts"
        out["notes"] = notes
        return out


    if claim_id == "C13" and parameter_sensitivity is not None and parameter_sensitivity_stats is not None:
        configs = set(parameter_sensitivity["config"].astype(str))
        pairwise = parameter_sensitivity_stats[parameter_sensitivity_stats["section"].astype(str) == "pairwise_vs_BASE"]
        supported = (
            configs == {"BASE", "LNS5", "LNS20", "REPAIR8", "REPAIR20"}
            and int(pd.to_numeric(parameter_sensitivity["runs"], errors="coerce").sum()) == 45
            and int(pd.to_numeric(parameter_sensitivity["hard_feasible_runs"], errors="coerce").sum()) == 45
            and len(pairwise) == 4
            and (pd.to_numeric(pairwise["p_holm"], errors="coerce") >= 0.05).all()
            and float(pd.to_numeric(parameter_sensitivity["max_abs_pct_delta_vs_base"], errors="coerce").max()) <= 2.31
        )
        max_delta = float(pd.to_numeric(parameter_sensitivity["max_abs_pct_delta_vs_base"], errors="coerce").max())
        min_holm = float(pd.to_numeric(pairwise["p_holm"], errors="coerce").min()) if len(pairwise) else float("nan")
        out["status"] = "supported_as_secondary_local_sensitivity" if supported else "not_supported_by_artifacts"
        out["notes"] = (
            f"secondary short-budget OAT screen: 45/45 hard-feasible; max absolute block-level score shift={max_delta:.3f}%; "
            f"minimum Holm-adjusted baseline-vs-neighbor p={min_holm:.3f}; scope is local robustness, not full-budget/global tuning."
        )
        return out


    if claim_id == "C14" and anytime_summary is not None and anytime_auc is not None and anytime_overall is not None:
        supported = False
        notes = "Anytime evidence is missing the expected locked design."
        if len(anytime_summary) == 12 and len(anytime_auc) == 6 and len(anytime_overall) == 1:
            r = anytime_overall.iloc[0]
            supported = (
                int(r["runs"]) == 48
                and int(r["hard_feasible_runs"]) == 48
                and int(r["strict_duty_runs"]) == 48
                and int(r["instance_cap_cells"]) == 12
                and int(r["ede_median_better_cells"]) == 12
                and int(r["paired_seed_cap_comparisons"]) == 24
                and int(r["ede_seed_wins"]) == 23
                and int(r["auc_trajectories"]) == 6
                and int(r["ede_better_auc_trajectories"]) == 6
                and float(r["median_auc_pct_reduction_ede_vs_stdde"]) > 0.0
                and float(r["max_runtime_over_cap_s"]) <= 0.10
                and (pd.to_numeric(anytime_summary["hard_feasible_runs"], errors="coerce") == 4).all()
                and (pd.to_numeric(anytime_summary["ede_median_better"], errors="coerce") == 1).all()
                and (pd.to_numeric(anytime_auc["ede_better_auc"], errors="coerce") == 1).all()
            )
            notes = (
                f"secondary short-budget anytime screen: 48/48 hard-feasible; "
                f"E-DE lower median J in {int(r['ede_median_better_cells'])}/{int(r['instance_cap_cells'])} instance-cap cells; "
                f"paired seed-cap wins={int(r['ede_seed_wins'])}/{int(r['paired_seed_cap_comparisons'])}; "
                f"lower log2-time normalized AUC in {int(r['ede_better_auc_trajectories'])}/{int(r['auc_trajectories'])} trajectories; "
                f"median AUC reduction={float(r['median_auc_pct_reduction_ede_vs_stdde']):.3f}%; "
                "scope is descriptive small-tier/S3 short-budget evidence, not a general convergence-rate or full-budget/tier-wide superiority claim."
            )
        out["status"] = "supported_as_secondary_anytime" if supported else "not_supported_by_artifacts"
        out["notes"] = notes
        return out

    out["status"] = "manual_review_needed"
    out["notes"] = "Claim id not wired into the submission generator."
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", nargs="?", default=str(TEMPLATES / "claim_evidence_map_template.csv"))
    parser.add_argument("--summary", default=str(GENERATED / "summary_by_method.csv"))
    parser.add_argument("--stats", default=str(GENERATED / "master_stats.csv"))
    parser.add_argument("--accepted-summary", default=None)
    parser.add_argument("--accepted-stats", default=None)
    parser.add_argument("--scenario-robustness", default=str(GENERATED / "robustness_by_scenario.csv"))
    parser.add_argument("--structure-robustness", default=str(GENERATED / "robustness_by_structure.csv"))
    parser.add_argument("--native-reductions", default=str(GENERATED / "tier_native_endpoint_reductions.csv"))
    parser.add_argument("--protocol-sensitivity", default=str(GENERATED / "protocol_sensitivity.csv"))
    parser.add_argument("--block-effect-ci", default=str(GENERATED / "block_effect_bootstrap_ci.csv"))
    parser.add_argument("--stability", default=str(GENERATED / "stability_by_method.csv"))
    parser.add_argument("--historical-diagnostics", default=str(GENERATED / "historical_variant_directional_diagnostics.csv"))
    parser.add_argument("--operational-shift-summary", default=str(GENERATED / "operational_shift_stress_summary.csv"))
    parser.add_argument("--parameter-sensitivity-summary", default=str(GENERATED / "parameter_sensitivity_summary.csv"))
    parser.add_argument("--parameter-sensitivity-stats", default=str(GENERATED / "parameter_sensitivity_stats.csv"))
    parser.add_argument("--anytime-summary", default=str(GENERATED / "anytime_profile_summary.csv"))
    parser.add_argument("--anytime-auc", default=str(GENERATED / "anytime_profile_auc.csv"))
    parser.add_argument("--anytime-overall", default=str(GENERATED / "anytime_profile_overall.csv"))
    parser.add_argument("--output", default=str(GENERATED / "claim_evidence_map.csv"))
    args = parser.parse_args()

    ensure_generated_dirs()
    summary_path = Path(args.summary)
    stats_path = Path(args.stats)
    accepted_summary_path = Path(args.accepted_summary) if args.accepted_summary else _default_accepted_path(summary_path)
    accepted_stats_path = Path(args.accepted_stats) if args.accepted_stats else _default_accepted_path(stats_path)

    template = pd.read_csv(Path(args.template))
    summary = _first_non_null(_load_optional_csv(accepted_summary_path), _load_optional_csv(summary_path))
    stats = _first_non_null(_load_optional_csv(accepted_stats_path), _load_optional_csv(stats_path))
    scenario = _load_optional_csv(Path(args.scenario_robustness))
    structure = _load_optional_csv(Path(args.structure_robustness))
    native = _load_optional_csv(Path(args.native_reductions))
    protocol = _load_optional_csv(Path(args.protocol_sensitivity))
    ci = _load_optional_csv(Path(args.block_effect_ci))
    stability = _load_optional_csv(Path(args.stability))
    historical = _load_optional_csv(Path(args.historical_diagnostics))
    operational = _load_optional_csv(Path(args.operational_shift_summary))
    parameter_sensitivity = _load_optional_csv(Path(args.parameter_sensitivity_summary))
    parameter_sensitivity_stats = _load_optional_csv(Path(args.parameter_sensitivity_stats))
    anytime_summary = _load_optional_csv(Path(args.anytime_summary))
    anytime_auc = _load_optional_csv(Path(args.anytime_auc))
    anytime_overall = _load_optional_csv(Path(args.anytime_overall))

    completed = template.apply(
        _fill_claim_row,
        axis=1,
        summary=summary,
        stats=stats,
        scenario=scenario,
        structure=structure,
        native=native,
        protocol=protocol,
        ci=ci,
        stability=stability,
        historical=historical,
        operational=operational,
        parameter_sensitivity=parameter_sensitivity,
        parameter_sensitivity_stats=parameter_sensitivity_stats,
        anytime_summary=anytime_summary,
        anytime_auc=anytime_auc,
        anytime_overall=anytime_overall,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
