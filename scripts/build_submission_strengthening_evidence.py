#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

PRIMARY_METHODS = ["EDE", "StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"]
COMPARATORS = ["StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"]
HISTORICAL_VARIANTS = ["A1_NoSeed", "A2_NoJDE", "A3_NoLNS"]
TIER_CAPS = {"small": 1800.0, "medium": 3600.0, "large": 7200.0}
TOLERANCE_SEC = 10.0
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20_260_914


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def block_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["instance_id", "scenario_id", "method_id"])["j_scaled_final"]
        .median()
        .unstack("method_id")[PRIMARY_METHODS]
    )


def infer(df: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = block_table(df)
    fr = friedmanchisquare(*[blocks[m].to_numpy() for m in PRIMARY_METHODS])
    kendalls_w = float(fr.statistic / (len(blocks) * (len(PRIMARY_METHODS) - 1)))
    rows = [{
        "scope": scope,
        "section": "omnibus",
        "metric": "j_scaled_final",
        "control_method": "",
        "compare_method": "",
        "blocks": len(blocks),
        "k_methods": len(PRIMARY_METHODS),
        "statistic": float(fr.statistic),
        "p_value": float(fr.pvalue),
        "kendalls_w": kendalls_w,
        "median_diff_ede_minus_compare": np.nan,
        "p_holm": np.nan,
    }]
    raw = []
    pvals = []
    for method in COMPARATORS:
        diff = blocks["EDE"] - blocks[method]
        test = wilcoxon(blocks["EDE"], blocks[method], zero_method="wilcox")
        raw.append((method, float(test.statistic), float(test.pvalue), float(diff.median())))
        pvals.append(float(test.pvalue))
    holm = multipletests(pvals, method="holm")[1]
    for (method, stat, p, med), p_holm in zip(raw, holm):
        rows.append({
            "scope": scope,
            "section": "pairwise",
            "metric": "j_scaled_final",
            "control_method": "EDE",
            "compare_method": method,
            "blocks": len(blocks),
            "k_methods": len(PRIMARY_METHODS),
            "statistic": stat,
            "p_value": p,
            "kendalls_w": np.nan,
            "median_diff_ede_minus_compare": med,
            "p_holm": float(p_holm),
        })
    return pd.DataFrame(rows), blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, type=Path)
    ap.add_argument("--full", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    primary = pd.read_csv(args.primary)
    full = pd.read_csv(args.full)

    if len(primary) != 10800 or set(primary["method_id"]) != set(PRIMARY_METHODS):
        raise SystemExit("Primary ledger is not the locked 10,800-row five-method set")
    if len(full) != 17280:
        raise SystemExit("Full historical ledger is not 17,280 rows")

    primary = primary.copy()
    primary["walltime_cap_s"] = primary["tier"].map(TIER_CAPS)
    if primary["walltime_cap_s"].isna().any():
        raise SystemExit("Unknown tier in primary ledger")
    primary["runtime_cap_violation"] = (
        primary["runtime_sec"] > primary["walltime_cap_s"] + TOLERANCE_SEC
    )
    deviations = primary[primary["runtime_cap_violation"]].copy()

    raw_stats, raw_blocks = infer(primary, "all_10800")
    compliant_stats, compliant_blocks = infer(
        primary[~primary["runtime_cap_violation"]], "cap_compliant_10795"
    )
    protocol = pd.concat([raw_stats, compliant_stats], ignore_index=True)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ci_rows = []
    for method in COMPARATORS:
        diff = (raw_blocks["EDE"] - raw_blocks[method]).to_numpy()
        n = len(diff)
        idx = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
        medians = np.median(diff[idx], axis=1)
        ci_rows.append({
            "compare_method": method,
            "blocks": n,
            "median_diff_ede_minus_compare": float(np.median(diff)),
            "bootstrap_ci95_low": float(np.quantile(medians, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(medians, 0.975)),
            "ede_better_blocks_pct": float(np.mean(diff < 0) * 100.0),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
        })
    ci = pd.DataFrame(ci_rows)

    grouped = primary.groupby(["instance_id", "scenario_id", "method_id"])["j_scaled_final"]
    stability = grouped.agg(
        mean="mean",
        median="median",
        std="std",
        q10=lambda x: x.quantile(0.10),
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        q90=lambda x: x.quantile(0.90),
    ).reset_index()
    stability["iqr"] = stability["q75"] - stability["q25"]
    stability["p90_p10"] = stability["q90"] - stability["q10"]
    stability["cv"] = stability["std"] / stability["mean"]
    stability_summary = (
        stability.groupby("method_id")
        .agg(
            blocks=("iqr", "size"),
            median_within_block_iqr=("iqr", "median"),
            median_within_block_p90_p10=("p90_p10", "median"),
            median_within_block_cv=("cv", "median"),
            median_within_block_std=("std", "median"),
        )
        .reset_index()
        .sort_values("median_within_block_iqr")
    )

    full_blocks = (
        full.groupby(["instance_id", "scenario_id", "method_id"])["j_scaled_final"]
        .median().unstack("method_id")
    )
    hist_rows = []
    for variant in HISTORICAL_VARIANTS:
        diff = full_blocks["EDE"] - full_blocks[variant]
        hist_rows.append({
            "variant": variant,
            "blocks": len(diff),
            "ede_global_block_median": float(full_blocks["EDE"].median()),
            "variant_global_block_median": float(full_blocks[variant].median()),
            "median_block_diff_ede_minus_variant": float(diff.median()),
            "ede_better_blocks_pct": float(np.mean(diff < 0) * 100.0),
            "interpretation": "exploratory_directional_only_not_causal",
        })
    historical = pd.DataFrame(hist_rows)

    files = {
        "protocol_sensitivity.csv": protocol,
        "runtime_cap_deviations.csv": deviations[
            ["run_id", "instance_id", "scenario_id", "method_id", "seed",
             "tier", "runtime_sec", "walltime_cap_s"]
        ],
        "block_effect_bootstrap_ci.csv": ci,
        "stability_by_method.csv": stability_summary,
        "historical_variant_directional_diagnostics.csv": historical,
    }
    for name, frame in files.items():
        frame.to_csv(args.out / name, index=False)

    provenance = {
        "schema": "ede-submission-strengthening-evidence-v1",
        "primary_source": {
            "path": str(args.primary),
            "rows": len(primary),
            "sha256": sha256(args.primary),
        },
        "full_historical_source": {
            "path": str(args.full),
            "rows": len(full),
            "sha256": sha256(args.full),
        },
        "runtime_cap_rule": {
            "caps_seconds": TIER_CAPS,
            "tolerance_seconds": TOLERANCE_SEC,
            "paired_exclusion": "all five primary methods for the same instance/scenario/seed",
        },
        "bootstrap": {
            "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "unit": "instance_id x scenario_id block median",
        },
        "outputs": {
            name: sha256(args.out / name)
            for name in files
        },
    }
    (args.out / "strengthening_evidence_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"strengthening_evidence=PASS out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
