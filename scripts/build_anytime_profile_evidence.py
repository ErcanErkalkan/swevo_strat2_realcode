#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_INSTANCES = ["C101", "R101", "RC101"]
EXPECTED_METHODS = ["EDE", "StdDE"]
EXPECTED_SEEDS = [1, 2]
EXPECTED_CAPS = [1, 2, 4, 8]
EXPECTED_SCENARIO = "S3_mixed_fleet_arc"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="data/secondary/anytime_profile_locked.csv")
    ap.add_argument("--out", default="results/primary")
    args = ap.parse_args()

    ledger = Path(args.ledger)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ledger)

    required = {
        "instance_id", "scenario_id", "method_id", "seed", "walltime_cap_s",
        "accepted_final", "strict_duty_final", "v_cap_final", "v_tw_final",
        "v_shift_final", "overtime_sum_final", "j_scaled_final", "runtime_sec",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    if len(df) != 48:
        raise SystemExit(f"expected 48 rows, found {len(df)}")
    if set(df["instance_id"].astype(str)) != set(EXPECTED_INSTANCES):
        raise SystemExit("instance grid mismatch")
    if set(df["method_id"].astype(str)) != set(EXPECTED_METHODS):
        raise SystemExit("method grid mismatch")
    if set(df["seed"].astype(int)) != set(EXPECTED_SEEDS):
        raise SystemExit("seed grid mismatch")
    if set(df["walltime_cap_s"].astype(int)) != set(EXPECTED_CAPS):
        raise SystemExit("walltime-cap grid mismatch")
    if set(df["scenario_id"].astype(str)) != {EXPECTED_SCENARIO}:
        raise SystemExit("scenario mismatch")
    if df[["instance_id", "method_id", "seed", "walltime_cap_s"]].duplicated().any():
        raise SystemExit("duplicate instance/method/seed/cap rows")
    if int(pd.to_numeric(df["accepted_final"]).sum()) != 48:
        raise SystemExit("not all runs accepted")
    if int(pd.to_numeric(df["strict_duty_final"]).sum()) != 48:
        raise SystemExit("not all runs strict duty")
    for col in ["v_cap_final", "v_tw_final", "v_shift_final", "overtime_sum_final"]:
        if float(pd.to_numeric(df[col], errors="coerce").abs().max()) > 1e-9:
            raise SystemExit(f"nonzero feasibility/duty value in {col}")
    runtime_over = pd.to_numeric(df["runtime_sec"], errors="coerce") - pd.to_numeric(df["walltime_cap_s"], errors="coerce")
    if float(runtime_over.max()) > 0.10:
        raise SystemExit(f"runtime exceeds cap by more than 0.10 s: max={float(runtime_over.max()):.6f}")

    p = df.pivot(index=["instance_id", "walltime_cap_s", "seed"], columns="method_id", values="j_scaled_final").sort_index()
    if p.shape != (24, 2) or p.isna().any().any():
        raise SystemExit(f"incomplete paired grid {p.shape}")

    rows = []
    for inst in EXPECTED_INSTANCES:
        for cap in EXPECTED_CAPS:
            sub = df[(df["instance_id"].astype(str) == inst) & (pd.to_numeric(df["walltime_cap_s"]) == cap)]
            med = sub.groupby("method_id")["j_scaled_final"].median()
            ede = float(med["EDE"])
            std = float(med["StdDE"])
            pair = p.loc[(inst, cap)]
            rows.append({
                "instance_id": inst,
                "walltime_cap_s": cap,
                "paired_seeds": 2,
                "hard_feasible_runs": int(pd.to_numeric(sub["accepted_final"]).sum()),
                "strict_duty_runs": int(pd.to_numeric(sub["strict_duty_final"]).sum()),
                "median_j_ede": ede,
                "median_j_stdde": std,
                "ede_median_better": int(ede < std),
                "median_pct_reduction_ede_vs_stdde": 100.0 * (std - ede) / std,
                "ede_seed_wins": int((pair["EDE"] < pair["StdDE"]).sum()),
                "ties": int(np.isclose(pair["EDE"], pair["StdDE"], rtol=0.0, atol=1e-12).sum()),
                "stdde_seed_wins": int((pair["EDE"] > pair["StdDE"]).sum()),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "anytime_profile_summary.csv", index=False)

    work = df.copy()
    work["log2_cap"] = np.log2(pd.to_numeric(work["walltime_cap_s"], errors="raise"))
    auc_rows = []
    for (inst, seed, method), g in work.groupby(["instance_id", "seed", "method_id"]):
        g = g.sort_values("log2_cap")
        x = g["log2_cap"].to_numpy(dtype=float)
        y = g["j_scaled_final"].to_numpy(dtype=float)
        auc = float(np.trapezoid(y, x) / (x.max() - x.min()))
        auc_rows.append({"instance_id": str(inst), "seed": int(seed), "method_id": str(method), "log2_time_normalized_auc": auc})
    auc_long = pd.DataFrame(auc_rows)
    auc_wide = auc_long.pivot(index=["instance_id", "seed"], columns="method_id", values="log2_time_normalized_auc").reset_index()
    auc_wide["ede_better_auc"] = (auc_wide["EDE"] < auc_wide["StdDE"]).astype(int)
    auc_wide["auc_pct_reduction_ede_vs_stdde"] = 100.0 * (auc_wide["StdDE"] - auc_wide["EDE"]) / auc_wide["StdDE"]
    auc_wide.to_csv(out / "anytime_profile_auc.csv", index=False)

    overall = pd.DataFrame([{
        "runs": 48,
        "hard_feasible_runs": int(pd.to_numeric(df["accepted_final"]).sum()),
        "strict_duty_runs": int(pd.to_numeric(df["strict_duty_final"]).sum()),
        "instance_cap_cells": 12,
        "ede_median_better_cells": int(summary["ede_median_better"].sum()),
        "paired_seed_cap_comparisons": 24,
        "ede_seed_wins": int((p["EDE"] < p["StdDE"]).sum()),
        "ties": int(np.isclose(p["EDE"], p["StdDE"], rtol=0.0, atol=1e-12).sum()),
        "stdde_seed_wins": int((p["EDE"] > p["StdDE"]).sum()),
        "auc_trajectories": 6,
        "ede_better_auc_trajectories": int(auc_wide["ede_better_auc"].sum()),
        "median_auc_pct_reduction_ede_vs_stdde": float(auc_wide["auc_pct_reduction_ede_vs_stdde"].median()),
        "max_runtime_over_cap_s": float(runtime_over.max()),
        "scope": "secondary_small_tier_s3_short_budget_descriptive_not_general_convergence",
    }])
    overall.to_csv(out / "anytime_profile_overall.csv", index=False)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Instance & Cap (s) & Median E-DE $J$ & Median Std. DE $J$ & Reduction (\%) & Paired W/T/L \\",
        r"\midrule",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"{r.instance_id} & {int(r.walltime_cap_s)} & {float(r.median_j_ede):.3f} & "
            f"{float(r.median_j_stdde):.3f} & {float(r.median_pct_reduction_ede_vs_stdde):.2f} & "
            f"{int(r.ede_seed_wins)}/{int(r.ties)}/{int(r.stdde_seed_wins)}" + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Secondary short-budget anytime profile on C101, R101, and RC101 under S3 with paired seeds 1--2. Lower $J$ is better. W/T/L denotes E-DE wins/ties/losses against Standard DE at the same instance, cap, and seed. This descriptive small-tier screen does not establish a general convergence rate or full-budget/tier-wide superiority.}",
        r"\label{tab:anytime_profile_secondary}",
        r"\end{table}",
    ]
    (out / "table_anytime_profile.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "schema": "ede-secondary-anytime-profile-v1",
        "input": {"path": ledger.as_posix(), "bytes": ledger.stat().st_size, "sha256": sha256(ledger)},
        "design": {
            "instances": EXPECTED_INSTANCES,
            "scenario": EXPECTED_SCENARIO,
            "methods": EXPECTED_METHODS,
            "seeds": EXPECTED_SEEDS,
            "walltime_caps_s": EXPECTED_CAPS,
            "eval_budget_guard": 1000,
            "scope": "secondary_small_tier_s3_short_budget_descriptive_not_general_convergence",
        },
        "outputs": {},
    }
    for name in ["anytime_profile_summary.csv", "anytime_profile_auc.csv", "anytime_profile_overall.csv", "table_anytime_profile.tex"]:
        q = out / name
        payload["outputs"][name] = {"bytes": q.stat().st_size, "sha256": sha256(q)}
    (out / "anytime_profile_evidence_provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "anytime_profile=PASS "
        f"runs=48 feasible=48 cells={int(summary['ede_median_better'].sum())}/12 "
        f"paired_wins={int((p['EDE'] < p['StdDE']).sum())}/24 "
        f"auc_wins={int(auc_wide['ede_better_auc'].sum())}/6 "
        f"median_auc_reduction={float(auc_wide['auc_pct_reduction_ede_vs_stdde'].median()):.3f}%"
    )


if __name__ == "__main__":
    main()
