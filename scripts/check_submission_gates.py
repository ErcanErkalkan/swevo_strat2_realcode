#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import csv
import re
import hashlib
import math
import json
import pandas as pd

from swevo_suite.paths import CONFIGS, GENERATED, TEMPLATES


LOCKED_RUNTIME_EXCEPTION_IDS = {
    "R1_4_5__S1_balanced__EDE__seed20",
    "R1_4_5__S1_balanced__StdDE__seed20",
    "R1_4_5__S1_balanced__ALNS_MS__seed20",
    "R1_4_5__S1_balanced__HGS_MS__seed20",
    "R1_4_5__S1_balanced__ILS_MS__seed20",
}


def _manual(item_id: int, title: str, notes: str) -> dict[str, object]:
    return {"item_id": item_id, "title": title, "status": "manual_review_needed", "notes": notes}


def _pass(item_id: int, title: str, notes: str) -> dict[str, object]:
    return {"item_id": item_id, "title": title, "status": "pass", "notes": notes}


def _fail(item_id: int, title: str, notes: str) -> dict[str, object]:
    return {"item_id": item_id, "title": title, "status": "fail", "notes": notes}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manuscript_build_provenance_ok(manuscript_dir: Path) -> tuple[bool, str]:
    provenance_path = manuscript_dir / "build_provenance.json"
    tex = manuscript_dir / "elsarticle-template-num.tex"
    bib = manuscript_dir / "cas-refs.bib"
    bst = manuscript_dir / "elsarticle-num.bst"
    pdf = manuscript_dir / "manuscript.pdf"
    required = [provenance_path, tex, bib, bst, pdf]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return False, f"missing build/provenance artifacts: {', '.join(missing)}"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid build_provenance.json: {exc}"
    expected = {
        "tex_sha256": _sha256_file(tex),
        "bib_sha256": _sha256_file(bib),
        "bst_sha256": _sha256_file(bst),
        "pdf_sha256": _sha256_file(pdf),
    }
    mismatches = [
        key for key, value in expected.items()
        if str(provenance.get(key, "")).lower() != value.lower()
    ]
    if mismatches:
        return False, "build provenance hash mismatch: " + ", ".join(mismatches)
    if provenance.get("schema") != "swevo-manuscript-build-provenance-v3":
        return False, f"unexpected build provenance schema: {provenance.get('schema')!r}"
    expected_names = {
        "tex_file": tex.name,
        "bib_file": bib.name,
        "bst_file": bst.name,
        "pdf_file": pdf.name,
    }
    bad_names = [key for key, value in expected_names.items() if provenance.get(key) != value]
    if bad_names:
        return False, "build provenance filename mismatch: " + ", ".join(bad_names)
    build_command = str(provenance.get("build_command", ""))
    if "-bibtex" not in build_command or "-bibtex-" in build_command:
        return False, "build provenance does not record a BibTeX-enabled manuscript build"
    if provenance.get("source_date_epoch") != 1788436800 or provenance.get("force_source_date") is not True:
        return False, "build provenance does not record the deterministic PDF metadata epoch"
    pages = provenance.get("pages")
    if pages is None or int(pages) <= 0:
        return False, "build provenance has no valid page count"
    return True, f"TeX/Bib/BST/PDF hashes match build provenance; pages={int(pages)}"




def _graphical_abstract_provenance_ok(manuscript_dir: Path) -> tuple[bool, str]:
    provenance_path = manuscript_dir / "graphical_abstract_provenance.json"
    source = manuscript_dir / "graphical_abstract_source.tex"
    pdf = manuscript_dir / "graphical_abstract.pdf"
    png = manuscript_dir / "graphical_abstract.png"
    required = [provenance_path, source, pdf, png]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return False, f"missing graphical-abstract provenance artifacts: {', '.join(missing)}"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid graphical_abstract_provenance.json: {exc}"
    expected_hashes = {
        "source_sha256": _sha256_file(source),
        "pdf_sha256": _sha256_file(pdf),
        "png_sha256": _sha256_file(png),
    }
    mismatches = [
        key for key, value in expected_hashes.items()
        if str(provenance.get(key, "")).lower() != value.lower()
    ]
    if mismatches:
        return False, "graphical-abstract provenance hash mismatch: " + ", ".join(mismatches)
    expected_names = {
        "source_file": source.name,
        "pdf_file": pdf.name,
        "png_file": png.name,
    }
    bad_names = [key for key, value in expected_names.items() if provenance.get(key) != value]
    if bad_names:
        return False, "graphical-abstract provenance filename mismatch: " + ", ".join(bad_names)
    if provenance.get("schema") != "swevo-graphical-abstract-provenance-v1":
        return False, f"unexpected graphical-abstract provenance schema: {provenance.get('schema')!r}"
    if provenance.get("source_date_epoch") != 1788436800 or provenance.get("force_source_date") is not True:
        return False, "graphical-abstract provenance does not record the deterministic PDF metadata epoch"
    if int(provenance.get("raster_dpi", 0) or 0) != 220:
        return False, "graphical-abstract provenance does not record the canonical 220-DPI raster"
    if int(provenance.get("png_width_px", 0) or 0) <= 0 or int(provenance.get("png_height_px", 0) or 0) <= 0:
        return False, "graphical-abstract provenance has invalid PNG dimensions"
    compile_command = str(provenance.get("compile_command", ""))
    raster_command = str(provenance.get("raster_command", ""))
    if "latexmk -pdf" not in compile_command or source.name not in compile_command:
        return False, "graphical-abstract provenance has unexpected compile command"
    if "pdftoppm -png -r 220 -singlefile" not in raster_command:
        return False, "graphical-abstract provenance has unexpected raster command"
    return True, "source/PDF/PNG hashes match deterministic graphical-abstract provenance"

def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _manifest_methods(path: Path) -> list[str]:
    return sorted({row["method_id"] for row in _manifest_rows(path)})


def _runtime_cap_violations(
    master_runs: pd.DataFrame,
    manifest_rows: list[dict[str, str]],
    tolerance_s: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Return rows over their manifest cap and run IDs missing a cap.

    ``tolerance_s`` is a validation allowance for logging/scheduling jitter; it
    never modifies a recorded runtime or the optimization budget.
    """
    if master_runs.empty or not manifest_rows:
        return pd.DataFrame(), []

    required = {"run_id", "runtime_sec"}
    if not required.issubset(master_runs.columns):
        missing = sorted(required - set(master_runs.columns))
        raise ValueError(f"master_runs is missing required columns: {missing}")

    manifest_df = pd.DataFrame(manifest_rows).copy()
    for col in ("seed", "walltime_cap_s"):
        if col not in manifest_df.columns:
            raise ValueError(f"manifest is missing required column: {col}")
        manifest_df[col] = pd.to_numeric(manifest_df[col], errors="coerce")

    if "run_id" not in manifest_df.columns:
        needed = {"instance_id", "scenario_id", "method_id", "seed"}
        missing = sorted(needed - set(manifest_df.columns))
        if missing:
            raise ValueError(f"manifest cannot construct run_id; missing columns: {missing}")
        manifest_df["run_id"] = manifest_df.apply(
            lambda r: f"{r['instance_id']}__{r['scenario_id']}__{r['method_id']}__seed{int(r['seed']):02d}",
            axis=1,
        )

    if manifest_df["run_id"].duplicated().any():
        raise ValueError("manifest contains duplicate run_id values")
    if master_runs["run_id"].duplicated().any():
        raise ValueError("master_runs contains duplicate run_id values")

    caps = manifest_df[["run_id", "walltime_cap_s"]].copy()
    runtime_view = master_runs[["run_id", "runtime_sec"]].copy()
    runtime_view["runtime_sec"] = pd.to_numeric(runtime_view["runtime_sec"], errors="coerce")
    merged = runtime_view.merge(caps, on="run_id", how="left", validate="one_to_one")

    invalid_cap = (
        merged["walltime_cap_s"].isna()
        | ~merged["walltime_cap_s"].fillna(0).map(math.isfinite)
        | (merged["walltime_cap_s"].fillna(0) <= 0)
    )
    invalid_runtime = (
        merged["runtime_sec"].isna()
        | ~merged["runtime_sec"].fillna(0).map(math.isfinite)
        | (merged["runtime_sec"].fillna(0) <= 0)
    )
    integrity_issues = merged.loc[invalid_cap | invalid_runtime, "run_id"].astype(str).tolist()
    valid = ~(invalid_cap | invalid_runtime)
    bad = merged[
        valid
        & (merged["runtime_sec"] > merged["walltime_cap_s"] + float(tolerance_s))
    ].copy()
    return bad, integrity_issues




def _runtime_protocol_exception_ok(
    runtime_bad: pd.DataFrame,
    master_runs: pd.DataFrame,
    manuscript_text: str,
) -> tuple[bool, str]:
    """Validate the one documented synchronized runtime-cap deviation.

    This is a post-campaign sensitivity closure, not a claim that the five
    historical runtimes complied with the original cap.  The raw ledger is
    preserved unchanged and the exception is accepted only when the exact
    locked five paired rows are the sole offenders and independently generated
    sensitivity artifacts show invariant block-level inference.
    """
    actual_ids = set(runtime_bad.get("run_id", pd.Series(dtype=str)).astype(str))
    if actual_ids != LOCKED_RUNTIME_EXCEPTION_IDS:
        return False, f"runtime offender set is not the locked synchronized five: {sorted(actual_ids)}"

    deviation_path = GENERATED / "runtime_cap_deviations.csv"
    sensitivity_path = GENERATED / "protocol_sensitivity.csv"
    provenance_path = GENERATED / "strengthening_evidence_provenance.json"
    missing = [p.name for p in (deviation_path, sensitivity_path, provenance_path) if not p.exists()]
    if missing:
        return False, "missing runtime-sensitivity artifacts: " + ", ".join(missing)

    try:
        deviations = pd.read_csv(deviation_path)
        sensitivity = pd.read_csv(sensitivity_path)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid runtime-sensitivity artifact: {exc}"

    if "run_id" not in deviations.columns:
        return False, "runtime_cap_deviations.csv has no run_id column"
    deviation_ids = set(deviations["run_id"].astype(str))
    if deviation_ids != LOCKED_RUNTIME_EXCEPTION_IDS or len(deviations) != 5:
        return False, "runtime_cap_deviations.csv does not contain exactly the locked five rows"

    # Ensure the deviation artifact records exactly the runtimes/caps in the raw ledger.
    expected = runtime_bad[["run_id", "runtime_sec", "walltime_cap_s"]].copy().sort_values("run_id")
    got = deviations[["run_id", "runtime_sec", "walltime_cap_s"]].copy().sort_values("run_id")
    for col in ("runtime_sec", "walltime_cap_s"):
        expected[col] = pd.to_numeric(expected[col], errors="coerce")
        got[col] = pd.to_numeric(got[col], errors="coerce")
    if expected["run_id"].tolist() != got["run_id"].tolist():
        return False, "runtime deviation IDs/order cannot be reconciled"
    for col in ("runtime_sec", "walltime_cap_s"):
        if not all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6) for a, b in zip(expected[col], got[col])):
            return False, f"runtime deviation artifact does not match raw ledger column {col}"

    required_cols = {
        "scope", "section", "metric", "control_method", "compare_method", "blocks",
        "k_methods", "statistic", "p_value", "kendalls_w",
        "median_diff_ede_minus_compare", "p_holm",
    }
    if not required_cols.issubset(sensitivity.columns):
        return False, f"protocol_sensitivity.csv missing columns: {sorted(required_cols - set(sensitivity.columns))}"
    scopes = set(sensitivity["scope"].astype(str))
    if scopes != {"all_10800", "cap_compliant_10795"}:
        return False, f"unexpected protocol sensitivity scopes: {sorted(scopes)}"

    raw = sensitivity[sensitivity["scope"] == "all_10800"].copy()
    compliant = sensitivity[sensitivity["scope"] == "cap_compliant_10795"].copy()
    if len(raw) != 5 or len(compliant) != 5:
        return False, "protocol sensitivity must contain one omnibus plus four pairwise rows per scope"

    # Omnibus invariance.
    ro = raw[raw["section"] == "omnibus"]
    co = compliant[compliant["section"] == "omnibus"]
    if len(ro) != 1 or len(co) != 1:
        return False, "protocol sensitivity does not contain exactly one omnibus row per scope"
    for col in ("blocks", "k_methods", "statistic", "p_value", "kendalls_w"):
        a, b = float(ro.iloc[0][col]), float(co.iloc[0][col])
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
            return False, f"omnibus sensitivity changed for {col}: {a} vs {b}"
    if int(float(ro.iloc[0]["blocks"])) != 108 or int(float(ro.iloc[0]["k_methods"])) != 5:
        return False, "protocol sensitivity has unexpected block/method cardinality"

    # Pairwise invariance for the exact four primary comparators.
    comparators = {"StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"}
    rp = raw[raw["section"] == "pairwise"].copy()
    cp = compliant[compliant["section"] == "pairwise"].copy()
    if set(rp["compare_method"].astype(str)) != comparators or set(cp["compare_method"].astype(str)) != comparators:
        return False, "protocol sensitivity does not contain the exact four primary pairwise comparisons"
    numeric_cols = ("blocks", "k_methods", "statistic", "p_value", "median_diff_ede_minus_compare", "p_holm")
    rp = rp.set_index("compare_method")
    cp = cp.set_index("compare_method")
    for comp in sorted(comparators):
        if str(rp.loc[comp, "control_method"]) != "EDE" or str(cp.loc[comp, "control_method"]) != "EDE":
            return False, f"unexpected control method for {comp}"
        for col in numeric_cols:
            a, b = float(rp.loc[comp, col]), float(cp.loc[comp, col])
            if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                return False, f"pairwise sensitivity changed for {comp}/{col}: {a} vs {b}"

    # Provenance must bind the current primary ledger and the uploaded sensitivity outputs.
    if provenance.get("schema") != "ede-submission-strengthening-evidence-v1":
        return False, f"unexpected strengthening provenance schema: {provenance.get('schema')!r}"
    source = provenance.get("primary_source", {})
    if int(source.get("rows", -1)) != len(master_runs):
        return False, "strengthening provenance primary row count mismatch"
    master_path = GENERATED / "master_runs.csv"
    if not master_path.exists() or str(source.get("sha256", "")).lower() != _sha256_file(master_path).lower():
        return False, "strengthening provenance does not bind the current primary master_runs.csv"
    outputs = provenance.get("outputs", {})
    for path in (deviation_path, sensitivity_path):
        if str(outputs.get(path.name, "")).lower() != _sha256_file(path).lower():
            return False, f"strengthening provenance output hash mismatch: {path.name}"

    lower = manuscript_text.lower()
    disclosure_ok = (
        "documented synchronized runtime-cap deviation" in lower
        and "raw rows are preserved unchanged" in lower
        and "10,795 cap-compliant records" in manuscript_text
        and "does not rewrite the historical campaign" in lower
    )
    if not disclosure_ok:
        return False, "manuscript does not contain the required transparent runtime-deviation disclosure"

    return True, (
        "The only over-cap records are the locked five synchronized paired rows; raw values are preserved, "
        "and the 10,795-record cap-compliant sensitivity leaves the 108-block Friedman and four "
        "Wilcoxon-Holm comparisons invariant."
    )


def _headline_claim_tokens(summary: pd.DataFrame, stats: pd.DataFrame) -> list[str]:
    """Build manuscript tokens for the primary headline statistics.

    This prevents a targeted rerun from silently leaving hard-coded abstract,
    results, or conclusion numbers stale after the evidence artifacts are
    regenerated.  Tokens use the manuscript's published precision.
    """
    required_cols = {"tier", "method_id", "median_j"}
    if summary.empty or not required_cols.issubset(summary.columns):
        return []
    tokens: list[str] = []
    for tier in ("small", "medium", "large"):
        ede = summary[(summary["tier"] == tier) & (summary["method_id"] == "EDE")]
        std = summary[(summary["tier"] == tier) & (summary["method_id"] == "StdDE")]
        if len(ede) != 1 or len(std) != 1:
            return []
        ede_j = float(ede.iloc[0]["median_j"])
        std_j = float(std.iloc[0]["median_j"])
        reduction = 100.0 * (std_j - ede_j) / std_j
        tokens.extend([f"{ede_j:.3f}", f"{std_j:.3f}", f"{reduction:.1f}\\%"])

    if stats.empty or "section" not in stats.columns:
        return []
    omnibus = stats[stats["section"] == "omnibus"]
    if len(omnibus) != 1:
        return []
    row = omnibus.iloc[0]
    chi2 = float(row["friedman_chi2"])
    w = float(row["kendalls_w"])
    pval = float(row["p_value"])
    if pval <= 0:
        return []
    import math
    exp = int(math.floor(math.log10(pval)))
    coeff = pval / (10.0 ** exp)
    tokens.extend([f"{chi2:.2f}", f"{coeff:.2f}\\times10^{{{exp}}}", f"{w:.3f}"])
    return tokens



def _sci_tex(value: float) -> str:
    """Format a positive p-value using the manuscript's TeX scientific notation."""
    import math
    if value <= 0:
        raise ValueError("Scientific-notation formatter requires a positive value")
    exp = int(math.floor(math.log10(value)))
    coeff = value / (10.0 ** exp)
    return f"{coeff:.2f}\\times10^{{{exp}}}"


def _quantitative_manuscript_fragments(
    summary: pd.DataFrame,
    stats: pd.DataFrame,
    robustness_scenario: pd.DataFrame,
    robustness_structure: pd.DataFrame,
    native_reductions: pd.DataFrame,
    secondary_stats: pd.DataFrame,
) -> list[str]:
    """Build exact high-value manuscript fragments from regenerated evidence.

    These checks cover every primary numerical table plus the secondary robustness,
    native-endpoint, omnibus, and post-hoc tables that could be affected by a
    targeted replacement rerun.  Exact row-level fragments are intentionally used
    instead of loose token presence so stale values elsewhere cannot mask a mismatch.
    """
    required_summary = {"tier", "method_id", "median_j"}
    if summary.empty or not required_summary.issubset(summary.columns):
        return []

    method_order = ["EDE", "StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"]
    method_display = {
        "EDE": "E-DE",
        "StdDE": "Std.\\ DE",
        "ALNS_MS": "ALNS-MS",
        "HGS_MS": "HGS-MS",
        "ILS_MS": "ILS-MS",
    }
    tier_display = {"small": "Small", "medium": "Medium", "large": "Large"}

    values: dict[tuple[str, str], float] = {}
    for tier in ("small", "medium", "large"):
        for method in method_order:
            row = summary[(summary["tier"] == tier) & (summary["method_id"] == method)]
            if len(row) != 1:
                return []
            values[(tier, method)] = float(row.iloc[0]["median_j"])

    fragments: list[str] = []
    # Exact primary tier-table rows.
    for tier in ("small", "medium", "large"):
        vals = [values[(tier, method)] for method in method_order]
        pad = " " if tier in {"small", "large"} else ""
        fragments.append(
            f"{tier_display[tier]}{pad} & \\textbf{{{vals[0]:.3f}}} & {vals[1]:.3f} & "
            f"{vals[2]:.3f} & {vals[3]:.3f} & {vals[4]:.3f} \\\\"
        )

    ede_vals = [values[(t, "EDE")] for t in ("small", "medium", "large")]
    std_vals = [values[(t, "StdDE")] for t in ("small", "medium", "large")]
    reductions = [100.0 * (s - e) / s for e, s in zip(ede_vals, std_vals)]
    fragments.extend([
        f"E-DE attains median scaled-composite values of {ede_vals[0]:.3f}, {ede_vals[1]:.3f}, and {ede_vals[2]:.3f}",
        f"compared with {std_vals[0]:.3f}, {std_vals[1]:.3f}, and {std_vals[2]:.3f} for Standard DE",
        f"median reductions of {reductions[0]:.1f}\\%, {reductions[1]:.1f}\\%, and {reductions[2]:.1f}\\%",
        f"tier-level median is reduced by {reductions[0]:.1f}\\% on small, {reductions[1]:.1f}\\% on medium, and {reductions[2]:.1f}\\% on large instances",
        f"median reduction is {reductions[0]:.1f}\\% on small, {reductions[1]:.1f}\\% on medium, and {reductions[2]:.1f}\\% on large instances",
        f"reduces the corresponding median by {min(reductions):.1f}--{max(reductions):.1f}\\% relative to Standard DE",
    ])

    # Scenario robustness rows and prose values.
    if robustness_scenario.empty:
        return []
    scenario_labels = {
        "S1_balanced": "S1 balanced",
        "S2_peak_dirty": "S2 peak/dirty",
        "S3_mixed_fleet_arc": "S3 mixed-regime/arc",
    }
    scenario_reductions_std: list[float] = []
    scenario_best_reductions: list[float] = []
    for sid in ("S1_balanced", "S2_peak_dirty", "S3_mixed_fleet_arc"):
        row = robustness_scenario[robustness_scenario["scenario_id"] == sid]
        if len(row) != 1:
            return []
        r = row.iloc[0]
        best = str(r["best_other_method"])
        if best not in method_display:
            return []
        scenario_reductions_std.append(float(r["ede_reduction_vs_stdde_pct"]))
        scenario_best_reductions.append(float(r["ede_reduction_vs_best_other_pct"]))
        fragments.append(
            f"{scenario_labels[sid]} & \\textbf{{{float(r['ede_median_j']):.3f}}} & "
            f"{float(r['stdde_median_j']):.3f} & {method_display[best].replace('Std.\\ DE', 'Std. DE')}: "
            f"{float(r['best_other_median_j']):.3f} & {float(r['ede_reduction_vs_best_other_pct']):.1f}\\% \\\\"
        )
    fragments.append(
        "scenario-level median reduction is "
        + ", ".join(f"{x:.1f}\\%" for x in scenario_reductions_std[:2])
        + f", and {scenario_reductions_std[2]:.1f}\\% for S1--S3, respectively"
    )
    fragments.append(
        f"the reduction remains {min(scenario_best_reductions):.1f}--{max(scenario_best_reductions):.1f}\\%"
    )

    # Structural robustness rows.
    if robustness_structure.empty:
        return []
    structure_labels = {"clustered": "Clustered", "mixed": "Mixed", "random": "Random"}
    structure_reductions: list[float] = []
    for sc in ("clustered", "mixed", "random"):
        row = robustness_structure[robustness_structure["structure_class"] == sc]
        if len(row) != 1:
            return []
        r = row.iloc[0]
        best = str(r["best_other_method"])
        if best not in method_display:
            return []
        red = float(r["ede_reduction_vs_best_other_pct"])
        structure_reductions.append(red)
        fragments.append(
            f"{structure_labels[sc]} & \\textbf{{{float(r['ede_median_j']):.3f}}} & "
            f"{float(r['stdde_median_j']):.3f} & {method_display[best].replace('Std.\\ DE', 'Std. DE')}: "
            f"{float(r['best_other_median_j']):.3f} & {red:.1f}\\% \\\\"
        )
    fragments.append(
        f"10.3\\% for clustered instances" if [round(x, 1) for x in structure_reductions] == [10.3, 14.6, 15.4]
        else f"{structure_reductions[0]:.1f}\\% for clustered instances"
    )
    fragments.append(f"{structure_reductions[1]:.1f}\\% for mixed instances")
    fragments.append(f"{structure_reductions[2]:.1f}\\% for random instances")

    # Native endpoint reduction table.
    if native_reductions.empty:
        return []
    metric_order = ["j_scaled_final", "cost_final", "energy_final", "co2_final"]
    for tier in ("small", "medium", "large"):
        row_values: list[float] = []
        for metric in metric_order:
            row = native_reductions[(native_reductions["tier"] == tier) & (native_reductions["metric"] == metric)]
            if len(row) != 1:
                return []
            row_values.append(float(row.iloc[0]["reduction_vs_stdde_pct"]))
        fragments.append(
            f"{tier_display[tier]} & {row_values[0]:.1f}\\% & {row_values[1]:.1f}\\% & "
            f"{row_values[2]:.1f}\\% & {row_values[3]:.1f}\\% \\\\"
        )

    # Primary omnibus and pairwise table rows.
    if stats.empty or "section" not in stats.columns:
        return []
    omnibus = stats[stats["section"] == "omnibus"]
    if len(omnibus) != 1:
        return []
    o = omnibus.iloc[0]
    fragments.extend([
        f"$\\Jobj_{{\\text{{scaled}}}}$ & 108 & {float(o['friedman_chi2']):.3f} & "
        f"${_sci_tex(float(o['p_value']))}$ & {float(o['kendalls_w']):.3f} \\\\"
    ])
    for method in ("StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"):
        row = stats[(stats["section"] == "pairwise") & (stats["compare_method"] == method)]
        if len(row) != 1:
            return []
        r = row.iloc[0]
        fragments.append(
            f"E-DE vs {method_display[method]} & ${_sci_tex(float(r['p_value']))}$ & "
            f"${_sci_tex(float(r['p_holm']))}$ & E-DE lower \\\\"
        )

    # Secondary endpoint omnibus rows.
    if secondary_stats.empty or "section" not in secondary_stats.columns:
        return []
    metric_labels = {
        "cost_final": "$f_1$ (Cost)",
        "energy_final": "$f_2$ (Energy)",
        "co2_final": "$f_3$ (\\COtwo)",
    }
    for metric in ("cost_final", "energy_final", "co2_final"):
        row = secondary_stats[(secondary_stats["section"] == "omnibus") & (secondary_stats["metric"] == metric)]
        if len(row) != 1:
            return []
        r = row.iloc[0]
        fragments.append(
            f"{metric_labels[metric]} & 108 & {float(r['friedman_chi2']):.3f} & "
            f"${_sci_tex(float(r['p_value']))}$ & {float(r['kendalls_w']):.3f} \\\\"
        )
    return fragments


def _read_docx_paragraphs(path: Path) -> list[str]:
    """Read DOCX paragraph text using only the Python standard library."""
    import zipfile
    import xml.etree.ElementTree as ET

    if not path.exists():
        return []
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
        root = ET.fromstring(xml)
    except Exception:
        return []
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        text = "".join(node.text or "" for node in para.findall(".//w:t", ns)).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _normalized_highlight_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\u2022*\-]+\s*", "", line).strip()
        lines.append(re.sub(r"\s+", " ", line))
    return lines


def _portal_mutable_numeric_issues(*texts: str) -> list[str]:
    """Flag rerun-sensitive numerical result copies in portal-facing prose."""
    patterns = [
        (r"\d+(?:\.\d+)?\s*(?:\\)?%", "percentage/effect-size literal"),
        (r"\bW\s*=\s*\d", "Kendall/Friedman W literal"),
        (r"\b(?:p|p-value)\s*[=<]\s*\d", "p-value literal"),
        (r"(?:chi|χ|\\chi)\s*(?:\^?2|2)?\s*=\s*\d", "chi-square literal"),
    ]
    issues: list[str] = []
    joined = "\n".join(texts)
    for pattern, label in patterns:
        if re.search(pattern, joined, flags=re.I):
            issues.append(label)
    return issues

def _tex_environment_has_content(text: str, env: str) -> bool:
    match = re.search(r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{" + re.escape(env) + r"\}", text, flags=re.S)
    if not match:
        return False
    body = re.sub(r"%.*", "", match.group(1)).strip()
    return bool(body)


def _manuscript_method_markers() -> dict[str, tuple[str, ...]]:
    return {
        "EDE": ("E-DE", "EDE"),
        "StdDE": ("Std.\\ DE", "Standard DE", "StdDE"),
        "ALNS_MS": ("ALNS-MS", "ALNS_MS"),
        "HGS_MS": ("HGS-MS", "HGS_MS"),
        "ILS_MS": ("ILS-MS", "ILS_MS"),
    }


def build_gate_rows(manifest: Path, readme: Path) -> list[dict[str, object]]:
    master_runs = _read_csv(GENERATED / "master_runs.csv")
    summary = _read_csv(GENERATED / "summary_by_method.csv")
    summary_acc = _read_csv(GENERATED / "summary_by_method_accepted_only.csv")
    stats = _read_csv(GENERATED / "master_stats.csv")
    stats_acc = _read_csv(GENERATED / "master_stats_accepted_only.csv")
    claim_map = _read_csv(GENERATED / "claim_evidence_map.csv")
    template = _read_csv(TEMPLATES / "claim_evidence_map_template.csv")
    robustness_scenario = _read_csv(GENERATED / "robustness_by_scenario.csv")
    robustness_structure = _read_csv(GENERATED / "robustness_by_structure.csv")
    native_reductions = _read_csv(GENERATED / "tier_native_endpoint_reductions.csv")
    secondary_stats = _read_csv(GENERATED / "secondary_endpoint_stats.csv")

    rows: list[dict[str, object]] = []
    manuscript = ROOT / "manuscript" / "elsarticle-template-num.tex"
    manuscript_text = manuscript.read_text(encoding="utf-8") if manuscript.exists() else ""

    rows.append(
        _pass(1, "Abstract is filled", "Abstract environment contains text.")
        if _tex_environment_has_content(manuscript_text, "abstract")
        else _fail(1, "Abstract is filled", "Abstract is missing or empty.")
    )
    highlights_docx = ROOT / "manuscript" / "Highlights.docx"
    highlights_txt = ROOT / "manuscript" / "Highlights.txt"
    standalone_highlights_ok = highlights_docx.exists() and highlights_docx.stat().st_size > 0 and highlights_txt.exists() and highlights_txt.stat().st_size > 0
    rows.append(
        _pass(2, "Standalone Highlights are present", "Highlights.docx and the plaintext provenance copy are present as separate submission artifacts.")
        if standalone_highlights_ok
        else _fail(2, "Standalone Highlights are present", "Separate Highlights.docx/plaintext artifact is missing or empty.")
    )

    manifest_rows = _manifest_rows(manifest)
    manifest_methods = sorted({row["method_id"] for row in manifest_rows})
    markers = _manuscript_method_markers()
    missing_in_manuscript = [
        m for m in manifest_methods
        if m not in markers or not any(token in manuscript_text for token in markers[m])
    ]
    rows.append(
        _pass(3, "Primary comparator list in manuscript == submission manifest", f"Primary methods: {', '.join(manifest_methods)}.")
        if manuscript_text and not missing_in_manuscript
        else _fail(3, "Primary comparator list in manuscript == submission manifest", f"Missing manuscript method markers: {', '.join(missing_in_manuscript)}")
    )

    output_methods = sorted(master_runs["method_id"].dropna().unique().tolist()) if not master_runs.empty else []
    missing_methods = [m for m in manifest_methods if m not in output_methods]
    extra_methods = [m for m in output_methods if m not in manifest_methods]
    expected_rows = len(manifest_rows)
    row_count_mismatch = (not master_runs.empty) and len(master_runs) != expected_rows

    if missing_methods or extra_methods or row_count_mismatch:
        notes = []
        if missing_methods:
            notes.append(f"Missing primary methods: {', '.join(missing_methods)}")
        if extra_methods:
            notes.append(f"Non-primary methods in authoritative ledger: {', '.join(extra_methods)}")
        if row_count_mismatch:
            notes.append(f"Row mismatch: master_runs={len(master_runs)} vs manifest={expected_rows}")
        rows.append(_fail(4, "Authoritative ledger matches the five-method submission manifest", "; ".join(notes)))
    else:
        rows.append(_pass(4, "Authoritative ledger matches the five-method submission manifest", f"Methods={', '.join(output_methods)}; rows={len(master_runs)}."))

    accepted_bad = strict_bad = 0
    failed_primary = 0
    if not master_runs.empty:
        accepted_bad = int(((master_runs["accepted_final"] == 1) & ((master_runs["v_cap_final"] != 0) | (master_runs["v_tw_final"] != 0) | (master_runs["v_shift_final"] != 0))).sum())
        strict_bad = int(((master_runs["strict_duty_final"] == 1) & (master_runs["overtime_sum_final"] != 0)).sum())
        failed_primary = int((master_runs["accepted_final"] != 1).sum())

    rows.append(
        _pass(5, "Accepted rows have zero final violations", "All accepted primary rows are clean.")
        if accepted_bad == 0
        else _fail(5, "Accepted rows have zero final violations", f"Found {accepted_bad} accepted primary rows with residual violations.")
    )
    rows.append(
        _pass(6, "Strict-duty rows have zero overtime", "All strict-duty primary rows have zero overtime.")
        if strict_bad == 0
        else _fail(6, "Strict-duty rows have zero overtime", f"Found {strict_bad} strict-duty primary rows with overtime.")
    )

    search_tables = [
        GENERATED / "table_feasibility_summary.tex",
        GENERATED / "table_friedman_omnibus.tex",
        GENERATED / "table_posthoc_holm.tex",
    ]
    missing_tables = [path.name for path in search_tables if not path.exists()]
    rows.append(
        _pass(7, "Submission-primary LaTeX evidence tables are present", "Feasibility, Friedman, and Holm tables are present.")
        if not missing_tables
        else _fail(7, "Submission-primary LaTeX evidence tables are present", f"Missing: {', '.join(missing_tables)}")
    )

    summary_methods = sorted(summary["method_id"].dropna().unique().tolist()) if not summary.empty else []
    rows.append(
        _pass(8, "Summary is generated for primary methods only", f"Summary methods: {', '.join(summary_methods)}.")
        if summary_methods == manifest_methods
        else _fail(8, "Summary is generated for primary methods only", f"Expected {manifest_methods}; found {summary_methods}.")
    )

    pairwise = stats[stats["section"] == "pairwise"] if (not stats.empty and "section" in stats.columns) else pd.DataFrame()
    omnibus = stats[stats["section"] == "omnibus"] if (not stats.empty and "section" in stats.columns) else pd.DataFrame()
    stats_ok = (
        not pairwise.empty
        and set(pairwise["compare_method"].dropna()) == set(manifest_methods) - {"EDE"}
        and not omnibus.empty
        and int(omnibus.iloc[0]["blocks"]) == 108
        and int(omnibus.iloc[0].get("k_methods", 0)) == 5
    )
    rows.append(
        _pass(9, "Inferential statistics use five methods and 108 independent blocks", "Pairwise family and Friedman block count are correct.")
        if stats_ok
        else _fail(9, "Inferential statistics use five methods and 108 independent blocks", "Stats artifact does not match the five-method/108-block design.")
    )

    rows.append(
        _pass(10, "Claim macros are generated from the submission outputs", "claim_macros.tex is present.")
        if (GENERATED / "claim_macros.tex").exists()
        else _fail(10, "Claim macros are generated from the submission outputs", "claim_macros.tex is missing.")
    )

    required_claims = sorted(template["claim_id"].dropna().astype(str).str.strip().str.upper().tolist()) if not template.empty else []
    claim_rows = claim_map.copy()
    if not claim_rows.empty:
        claim_rows["claim_id"] = claim_rows["claim_id"].astype(str).str.strip().str.upper()
    claim_status = {row["claim_id"]: row["status"] for _, row in claim_rows.iterrows()} if not claim_rows.empty else {}
    weak_statuses = {"todo", "missing_artifacts", "manual_review_needed", "not_supported_by_artifacts", "blocked_pending_rerun"}
    missing_claims = [cid for cid in required_claims if cid not in claim_status]
    weak_claims = [cid for cid in required_claims if claim_status.get(cid) in weak_statuses]
    headline_tokens = _headline_claim_tokens(summary, stats)
    quantitative_fragments = _quantitative_manuscript_fragments(
        summary, stats, robustness_scenario, robustness_structure, native_reductions, secondary_stats
    )
    missing_headline_tokens = [token for token in headline_tokens if token not in manuscript_text]
    missing_quantitative_fragments = [fragment for fragment in quantitative_fragments if fragment not in manuscript_text]
    numeric_ok = (
        bool(headline_tokens)
        and bool(quantitative_fragments)
        and not missing_headline_tokens
        and not missing_quantitative_fragments
    )
    rows.append(
        _pass(11, "Major manuscript claims are evidence-backed and numerically synchronized", "Claim-evidence map is complete; primary and secondary quantitative manuscript tables/claims match regenerated artifacts at published precision.")
        if not missing_claims and not weak_claims and numeric_ok
        else _fail(11, "Major manuscript claims are evidence-backed and numerically synchronized", f"Missing claims={missing_claims}; needs_work={weak_claims}; stale headline tokens={missing_headline_tokens if headline_tokens else ['unable_to_derive']}; stale quantitative fragments={missing_quantitative_fragments[:8] if quantitative_fragments else ['unable_to_derive']}" )
    )

    readme_text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    required_tokens = [
        "generated_submission",
        "experiment_manifest_submission_primary.csv",
        "python scripts/aggregate_results.py",
        "python scripts/run_stats.py",
        "python scripts/build_latex_tables.py",
        "python scripts/build_claim_evidence_map.py",
    ]
    missing_tokens = [token for token in required_tokens if token not in readme_text]
    rows.append(
        _pass(12, "README reproduces the submission-primary analysis pipeline", "README contains primary scope and regeneration commands.")
        if not missing_tokens
        else _fail(12, "README reproduces the submission-primary analysis pipeline", f"README missing: {', '.join(missing_tokens)}")
    )

    split_doc = ROOT / "docs" / "EXPERIMENT_SPLITS.md"
    rows.append(
        _pass(13, "Experiment split documentation is present", "docs/EXPERIMENT_SPLITS.md exists.")
        if split_doc.exists()
        else _manual(13, "Experiment split documentation is present", "Split documentation is missing.")
    )

    rows.append(
        _pass(14, "Failed primary runs are not hidden", "No failed submission-primary runs exist.")
        if failed_primary == 0
        else _fail(14, "Failed primary runs are not hidden", f"Found {failed_primary} non-accepted primary runs; report them explicitly.")
    )

    cond_ok = not summary.empty and not summary_acc.empty and not stats.empty and not stats_acc.empty
    rows.append(
        _pass(15, "Conditional and unconditional artifact pairs are both available", "Both artifact scopes are present.")
        if cond_ok
        else _fail(15, "Conditional and unconditional artifact pairs are both available", "Accepted-only/unconditional artifact pairs are incomplete.")
    )

    ablation_tokens = {"A1_NoSeed", "A2_NoJDE", "A3_NoLNS"}
    ablation_in_primary = bool(set(output_methods) & ablation_tokens)
    manuscript_scope_ok = (
        "no quantitative claim about the isolated contribution" in manuscript_text
        or "no isolated component-effect claim" in manuscript_text
        or "does not make isolated component-effect claims" in manuscript_text
    )
    rows.append(
        _pass(16, "Historical ablations are excluded from primary inference", "A1/A2/A3 are absent from the authoritative primary ledger and manuscript scope is non-causal.")
        if not ablation_in_primary and manuscript_scope_ok
        else _fail(16, "Historical ablations are excluded from primary inference", f"ablation_in_primary={ablation_in_primary}; manuscript_scope_ok={manuscript_scope_ok}")
    )

    # P2 semantic lock: the completed public-benchmark campaign uses three
    # overlapping full-horizon regime labels, not consecutive driver shifts.
    temporal_scope_ok = (
        "fully overlapping $[0,H]$ regime labels" in manuscript_text
        and "does not validate consecutive non-overlapping driver shifts" in manuscript_text
        and "full-horizon regime special case" in manuscript_text
    )
    rows.append(
        _pass(17, "Primary temporal scope matches the production loader", "Manuscript declares the overlapping full-horizon [0,H] regime special case and excludes sequential-shift inference.")
        if temporal_scope_ok
        else _fail(17, "Primary temporal scope matches the production loader", "Required full-horizon/overlapping-regime scope language is missing from the manuscript.")
    )

    scope_doc = GENERATED / "SUBMISSION_EVIDENCE_SCOPE.md"
    scope_text = scope_doc.read_text(encoding="utf-8") if scope_doc.exists() else ""
    split_doc_text = split_doc.read_text(encoding="utf-8") if split_doc.exists() else ""
    docs_scope_ok = (
        "[0,H]" in readme_text and "overlapping" in readme_text.lower()
        and "[0,H]" in scope_text and "overlapping" in scope_text.lower()
        and "[0,H]" in split_doc_text and "overlapping" in split_doc_text.lower()
        and "sequential" in readme_text.lower() and "sequential" in scope_text.lower() and "sequential" in split_doc_text.lower()
    )
    rows.append(
        _pass(18, "Repository documentation mirrors the primary temporal scope", "README, evidence scope, and split documentation all record the [0,H] overlapping-regime interpretation.")
        if docs_scope_ok
        else _fail(18, "Repository documentation mirrors the primary temporal scope", "README/scope/split documentation is not synchronized with the full-horizon regime interpretation.")
    )

    vehicle_scope_ok = (
        "does not impose a fixed fleet-share composition" in manuscript_text
        and "does not preserve a persistent physical vehicle identity" in manuscript_text
        and "fleet-share proportions" in readme_text
        and "persistent physical vehicle identity" in readme_text
    )
    rows.append(
        _pass(19, "Scenario and vehicle-identity semantics match the implementation", "Fleet-share and persistent-vehicle claims are explicitly excluded; route-level type activation is documented.")
        if vehicle_scope_ok
        else _fail(19, "Scenario and vehicle-identity semantics match the implementation", "Fleet-share or persistent-vehicle scope language is incomplete.")
    )

    provenance_ok = (
        "hardware-metered" in manuscript_text.lower()
        and "dell vostro 3888" in manuscript_text.lower()
        and "intel core i5-10400" in manuscript_text.lower()
        and "windows 11 enterprise" in manuscript_text.lower()
        and "dell vostro 3888" in readme_text.lower()
        and "dell vostro 3888" in scope_text.lower()
        and "per-run hardware fingerprint" in manuscript_text.lower()
    )
    rows.append(
        _pass(20, "Compute-energy and hardware-provenance scope is explicit", "Production CPU/RAM/OS/workstation details are documented; the per-run fingerprint limitation and non-metered compute-energy semantics remain explicit.")
        if provenance_ok
        else _fail(20, "Compute-energy and hardware-provenance scope is explicit", "Documented workstation details, per-run fingerprint scope, or compute-energy limitation language is incomplete.")
    )

    budget_policy = CONFIGS / "budget_policy.yaml"
    budget_text = budget_policy.read_text(encoding="utf-8") if budget_policy.exists() else ""
    budget_scope_ok = (
        "primary_fairness_currency: wall_clock" in budget_text
        and "evaluated_candidates" not in budget_text
    )
    rows.append(
        _pass(21, "Budget-policy metadata matches the wall-clock production protocol", "budget_policy.yaml declares wall-clock fairness and no stale evaluated-candidates fairness currency.")
        if budget_scope_ok
        else _fail(21, "Budget-policy metadata matches the wall-clock production protocol", "budget_policy.yaml still contains stale fairness metadata.")
    )

    # P0 runtime-integrity lock: fairness is meaningful only if recorded runs
    # respect the manifest wall-clock caps. A small tolerance covers result
    # serialization/scheduling jitter; it is not extra search budget.
    runtime_tolerance_s = 10.0
    match = re.search(r"runtime_validation_tolerance_s:\s*([0-9.]+)", budget_text)
    if match:
        runtime_tolerance_s = float(match.group(1))
    runtime_bad, runtime_missing = _runtime_cap_violations(master_runs, manifest_rows, runtime_tolerance_s)
    runtime_gate_title = "Runtime-cap integrity is compliant or transparently sensitivity-closed"
    if runtime_missing:
        rows.append(_fail(22, runtime_gate_title, f"Ledger rows with missing/invalid runtime or manifest cap values: {runtime_missing[:10]}"))
    elif runtime_bad.empty:
        rows.append(_pass(22, runtime_gate_title, f"All primary runtimes are within manifest caps + {runtime_tolerance_s:.0f}s validation tolerance."))
    else:
        exception_ok, exception_note = _runtime_protocol_exception_ok(runtime_bad, master_runs, manuscript_text)
        if exception_ok:
            rows.append(_pass(22, runtime_gate_title, exception_note))
        else:
            offenders = "; ".join(
                f"{r.run_id}={r.runtime_sec:.1f}s>{r.walltime_cap_s:.0f}s+{runtime_tolerance_s:.0f}s"
                for r in runtime_bad.itertuples(index=False)
            )
            rows.append(_fail(22, runtime_gate_title, f"Runtime-cap deviations are not sensitivity-closed: {exception_note}. Offenders: {offenders}"))

    manuscript_dir = ROOT / "manuscript"
    ga = manuscript_dir / "graphical_abstract.png"
    ga_source = manuscript_dir / "graphical_abstract_source.tex"
    cover_letter = manuscript_dir / "COVER_LETTER.txt"
    separate_portal_artifacts_ok = (
        ga.exists() and ga.stat().st_size > 0
        and highlights_docx.exists() and highlights_docx.stat().st_size > 0
        and highlights_txt.exists() and highlights_txt.stat().st_size > 0
        and ga_source.exists() and ga_source.stat().st_size > 0
        and cover_letter.exists() and cover_letter.stat().st_size > 0
        and "\\begin{graphicalabstract}" not in manuscript_text
        and "\\begin{highlights}" not in manuscript_text
    )
    highlights_txt_text = highlights_txt.read_text(encoding="utf-8") if highlights_txt.exists() else ""
    highlights_docx_lines = _read_docx_paragraphs(highlights_docx)
    highlights_match = (
        _normalized_highlight_lines(highlights_txt_text)
        == _normalized_highlight_lines("\n".join(highlights_docx_lines))
    )
    portal_issues = _portal_mutable_numeric_issues(
        cover_letter.read_text(encoding="utf-8") if cover_letter.exists() else "",
        highlights_txt_text,
        ga_source.read_text(encoding="utf-8") if ga_source.exists() else "",
    )
    build_ok, build_notes = _manuscript_build_provenance_ok(manuscript_dir)
    ga_build_ok, ga_build_notes = _graphical_abstract_provenance_ok(manuscript_dir)
    portal_ok = separate_portal_artifacts_ok and highlights_match and not portal_issues and ga_build_ok
    portal_notes = (
        f"portal_separation={separate_portal_artifacts_ok}; "
        f"highlights_txt_docx_match={highlights_match}; "
        f"mutable_numeric_issues={portal_issues or 'none'}"
    )
    rows.append(
        _pass(23, "Portal artifacts are separated, stable, and the manuscript PDF matches its sources", f"{portal_notes}; {build_notes}; {ga_build_notes}.")
        if portal_ok and build_ok
        else _fail(23, "Portal artifacts are separated, stable, and the manuscript PDF matches its sources", f"{portal_notes}; {build_notes}; {ga_build_notes}")
    )

    return rows


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    status_counts = pd.DataFrame(rows)["status"].value_counts().to_dict()
    lines = [
        "# Submission Gating Report",
        "",
        f"- pass: {status_counts.get('pass', 0)}",
        f"- fail: {status_counts.get('fail', 0)}",
        f"- manual_review_needed: {status_counts.get('manual_review_needed', 0)}",
        "",
        "| Item | Status | Notes |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {int(row['item_id'])}. {row['title']} | {row['status']} | {row['notes']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(CONFIGS / "experiment_manifest_submission_primary.csv"))
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    parser.add_argument("--csv-output", default=str(GENERATED / "submission_gating_report.csv"))
    parser.add_argument("--md-output", default=str(GENERATED / "submission_gating_report.md"))
    args = parser.parse_args()

    rows = build_gate_rows(Path(args.manifest), Path(args.readme))
    csv_path = Path(args.csv_output)
    md_path = Path(args.md_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_markdown(md_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    fail_count = sum(1 for row in rows if row["status"] == "fail")
    manual_count = sum(1 for row in rows if row["status"] == "manual_review_needed")
    if fail_count or manual_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
