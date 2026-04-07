from __future__ import annotations
import math
from itertools import combinations
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

def load_runs(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def summary_by_method(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["phase", "tier", "method_id"], dropna=False)
          .agg(
              runs=("run_id", "count"),
              accepted_rate=("accepted_final", "mean"),
              strict_duty_rate=("strict_duty_final", "mean"),
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
    agg["accepted_rate"] *= 100.0
    agg["strict_duty_rate"] *= 100.0
    return agg

def pairwise_wilcoxon(df: pd.DataFrame, metric: str = "j_scaled_final", control_method: str = "EDE") -> pd.DataFrame:
    blocks = []
    methods = sorted(df["method_id"].unique())
    others = [m for m in methods if m != control_method]
    for method in others:
        merged = pd.merge(
            df[df["method_id"] == control_method][["instance_id","scenario_id","seed",metric]],
            df[df["method_id"] == method][["instance_id","scenario_id","seed",metric]],
            on=["instance_id","scenario_id","seed"],
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
    return out

def friedman_by_instance(df: pd.DataFrame, metric: str = "j_scaled_final") -> pd.DataFrame:
    per_instance = (
        df.groupby(["instance_id", "scenario_id", "method_id"])[metric]
          .median()
          .reset_index()
    )
    pivot = per_instance.pivot_table(index=["instance_id", "scenario_id"], columns="method_id", values=metric)
    pivot = pivot.dropna()
    methods = list(pivot.columns)
    if len(methods) < 3 or pivot.empty:
        return pd.DataFrame([{"metric": metric, "blocks": 0, "friedman_chi2": np.nan, "p_value": np.nan, "kendalls_w": np.nan}])
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
    }])
