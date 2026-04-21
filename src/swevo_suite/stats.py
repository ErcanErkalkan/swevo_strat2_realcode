from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None

SUMMARY_GROUP_COLS = ["phase", "tier", "method_id"]
PAIR_BLOCK_COLS = ["instance_id", "scenario_id", "seed"]


def load_runs(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _truthy_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes", "y", "t"})


def filter_accepted(df: pd.DataFrame) -> pd.DataFrame:
    return df[_truthy_mask(df["accepted_final"])].copy()


def summary_by_method(
    df: pd.DataFrame,
    *,
    accepted_only: bool = False,
    summary_scope: str | None = None,
) -> pd.DataFrame:
    working = df.copy()
    working["_accepted_mask"] = _truthy_mask(working["accepted_final"])
    working["_strict_mask"] = _truthy_mask(working["strict_duty_final"])
    base = (
        working.groupby(SUMMARY_GROUP_COLS, dropna=False)
        .agg(
            source_runs=("run_id", "count"),
            accepted_runs=("_accepted_mask", "sum"),
            strict_duty_runs=("_strict_mask", "sum"),
        )
        .reset_index()
    )
    base["accepted_rate"] = 100.0 * base["accepted_runs"] / base["source_runs"].clip(lower=1)
    base["strict_duty_rate"] = 100.0 * base["strict_duty_runs"] / base["source_runs"].clip(lower=1)

    perf_source = filter_accepted(working) if accepted_only else working
    perf = (
        perf_source.groupby(SUMMARY_GROUP_COLS, dropna=False)
        .agg(
            runs=("run_id", "count"),
            median_j=("j_scaled_final", "median"),
            median_cost=("cost_final", "median"),
            median_energy=("energy_final", "median"),
            median_co2=("co2_final", "median"),
            median_runtime=("runtime_sec", "median"),
            median_wh=("compute_wh", "median"),
            median_imp_per_wh=("imp_per_wh", "median"),
        )
        .reset_index()
    )
    out = base.merge(perf, on=SUMMARY_GROUP_COLS, how="left")
    out["runs"] = out["runs"].fillna(0).astype(int)
    out.insert(
        len(SUMMARY_GROUP_COLS),
        "summary_scope",
        summary_scope or ("accepted_only" if accepted_only else "unconditional"),
    )
    return out


def pairwise_wilcoxon(
    df: pd.DataFrame,
    metric: str = "j_scaled_final",
    control_method: str = "EDE",
    *,
    accepted_only: bool = False,
) -> pd.DataFrame:
    working = filter_accepted(df) if accepted_only else df.copy()
    blocks = []
    methods = sorted(working["method_id"].dropna().unique())
    others = [m for m in methods if m != control_method]
    for method in others:
        merged = pd.merge(
            working[working["method_id"] == control_method][PAIR_BLOCK_COLS + [metric]],
            working[working["method_id"] == method][PAIR_BLOCK_COLS + [metric]],
            on=PAIR_BLOCK_COLS,
            suffixes=("_ctrl","_cmp")
        ).dropna()
        if merged.empty:
            continue
        x = merged[f"{metric}_ctrl"].to_numpy()
        y = merged[f"{metric}_cmp"].to_numpy()
        diff = x - y
        if np.allclose(diff, 0.0):
            stat, p = 0.0, 1.0
        elif scipy_stats is None:
            p = np.nan
            stat = np.nan
        else:
            stat, p = scipy_stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        wins = np.sum(diff < 0)
        losses = np.sum(diff > 0)
        n = max(wins + losses, 1)
        rank_biserial = (wins - losses) / n
        blocks.append({
            "metric": metric,
            "control_method": control_method,
            "compare_method": method,
            "n_pairs": len(merged),
            "wilcoxon_stat": stat,
            "p_value": p,
            "median_diff_control_minus_compare": float(np.median(diff)),
            "rank_biserial": float(rank_biserial),
        })
    out = pd.DataFrame(blocks)
    if not out.empty and multipletests is not None and out["p_value"].notna().any():
        mask = out["p_value"].notna()
        adjusted = multipletests(out.loc[mask, "p_value"], method="holm")[1]
        out.loc[mask, "p_holm"] = adjusted
    else:
        out["p_holm"] = np.nan
    if not out.empty:
        out["summary_scope"] = "accepted_only" if accepted_only else "unconditional"
    return out


def friedman_by_instance(
    df: pd.DataFrame,
    metric: str = "j_scaled_final",
    *,
    accepted_only: bool = False,
) -> pd.DataFrame:
    working = filter_accepted(df) if accepted_only else df.copy()
    per_instance = (
        working.groupby(["instance_id", "scenario_id", "method_id"])[metric]
        .median()
        .reset_index()
    )
    pivot = per_instance.pivot_table(index=["instance_id", "scenario_id"], columns="method_id", values=metric)
    pivot = pivot.dropna()
    methods = list(pivot.columns)
    if len(methods) < 3 or pivot.empty:
        return pd.DataFrame([{
            "metric": metric,
            "blocks": 0,
            "friedman_chi2": np.nan,
            "p_value": np.nan,
            "kendalls_w": np.nan,
            "summary_scope": "accepted_only" if accepted_only else "unconditional",
        }])
    arrays = [pivot[m].to_numpy() for m in methods]
    if scipy_stats is None:
        chi2 = p = np.nan
    else:
        chi2, p = scipy_stats.friedmanchisquare(*arrays)
    n = len(pivot)
    k = len(methods)
    kendalls_w = chi2 / (n * (k - 1)) if n > 0 else np.nan
    return pd.DataFrame([{
        "metric": metric,
        "blocks": int(n),
        "k_methods": int(k),
        "friedman_chi2": float(chi2) if chi2 == chi2 else np.nan,
        "p_value": float(p) if p == p else np.nan,
        "kendalls_w": float(kendalls_w) if kendalls_w == kendalls_w else np.nan,
        "summary_scope": "accepted_only" if accepted_only else "unconditional",
    }])
