from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_claim_evidence_map.py"


def _load_script_module():
    spec = spec_from_file_location("build_claim_evidence_map", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_claim_evidence_map_marks_supported_claims(tmp_path: Path) -> None:
    module = _load_script_module()

    template = pd.DataFrame(
        [
            {
                "claim_id": "C01",
                "manuscript_section": "Results/Feasibility",
                "exact_claim_text": "E-DE achieved X% accepted hard-feasible terminations",
                "source_table_or_figure": "tab:feasibility_summary_generated",
                "source_csv_or_tex": "generated/summary_by_method.csv",
                "generation_script": "scripts/aggregate_results.py",
                "reproduction_command": "python scripts/aggregate_results.py",
                "status": "todo",
                "notes": "",
            },
            {
                "claim_id": "C02",
                "manuscript_section": "Results/Stats",
                "exact_claim_text": "E-DE significantly outperformed baseline on J_scaled",
                "source_table_or_figure": "tab:posthoc_holm_generated",
                "source_csv_or_tex": "generated/master_stats.csv",
                "generation_script": "scripts/run_stats.py",
                "reproduction_command": "python scripts/run_stats.py",
                "status": "todo",
                "notes": "",
            },
            {
                "claim_id": "C03",
                "manuscript_section": "Results/Omnibus",
                "exact_claim_text": "Across-instance Friedman analysis supported method differences",
                "source_table_or_figure": "tab:friedman_omnibus_generated",
                "source_csv_or_tex": "generated/master_stats.csv",
                "generation_script": "scripts/run_stats.py",
                "reproduction_command": "python scripts/run_stats.py",
                "status": "todo",
                "notes": "",
            },
            {
                "claim_id": "C04",
                "manuscript_section": "Results/Compute",
                "exact_claim_text": "E-DE improved improvement-per-Wh over StdDE",
                "source_table_or_figure": "generated/summary_by_method.csv",
                "source_csv_or_tex": "generated/summary_by_method.csv",
                "generation_script": "scripts/aggregate_results.py",
                "reproduction_command": "python scripts/aggregate_results.py",
                "status": "todo",
                "notes": "",
            },
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "phase": "main",
                "tier": "medium",
                "method_id": "EDE",
                "summary_scope": "accepted_only",
                "source_runs": 18,
                "accepted_runs": 18,
                "strict_duty_runs": 18,
                "accepted_rate": 100.0,
                "strict_duty_rate": 100.0,
                "runs": 18,
                "median_j": 0.88,
                "median_cost": 1.0,
                "median_energy": 1.0,
                "median_co2": 1.0,
                "median_runtime": 60.0,
                "median_wh": 1.43,
                "median_imp_per_wh": 0.056,
            },
            {
                "phase": "main",
                "tier": "medium",
                "method_id": "StdDE",
                "summary_scope": "accepted_only",
                "source_runs": 18,
                "accepted_runs": 18,
                "strict_duty_runs": 18,
                "accepted_rate": 100.0,
                "strict_duty_rate": 100.0,
                "runs": 18,
                "median_j": 1.06,
                "median_cost": 1.0,
                "median_energy": 1.0,
                "median_co2": 1.0,
                "median_runtime": 60.0,
                "median_wh": 1.44,
                "median_imp_per_wh": 0.015,
            },
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "metric": "j_scaled_final",
                "control_method": "EDE",
                "compare_method": "StdDE",
                "n_pairs": 18,
                "wilcoxon_stat": 0.0,
                "p_value": 0.00001,
                "median_diff_control_minus_compare": -0.16,
                "rank_biserial": 1.0,
                "p_holm": 0.00002,
                "summary_scope": "accepted_only",
                "section": "pairwise",
                "blocks": None,
                "k_methods": None,
                "friedman_chi2": None,
                "kendalls_w": None,
            },
            {
                "metric": "j_scaled_final",
                "control_method": None,
                "compare_method": None,
                "n_pairs": None,
                "wilcoxon_stat": None,
                "p_value": 0.000001,
                "median_diff_control_minus_compare": None,
                "rank_biserial": None,
                "p_holm": None,
                "summary_scope": "accepted_only",
                "section": "omnibus",
                "blocks": 18,
                "k_methods": 3,
                "friedman_chi2": 29.77,
                "kendalls_w": 0.82,
            },
        ]
    )

    completed = template.apply(module._fill_claim_row, axis=1, summary=summary, stats=stats)

    statuses = completed.set_index("claim_id")["status"].to_dict()
    assert statuses == {
        "C01": "supported_by_artifacts",
        "C02": "supported_by_artifacts",
        "C03": "supported_by_artifacts",
        "C04": "supported_by_artifacts",
    }


def test_claim_evidence_map_normalizes_claim_id_case_and_whitespace() -> None:
    module = _load_script_module()

    template = pd.DataFrame(
        [
            {
                "claim_id": " c01 ",
                "manuscript_section": "Results/Feasibility",
                "exact_claim_text": "E-DE achieved X% accepted hard-feasible terminations",
                "source_table_or_figure": "tab:feasibility_summary_generated",
                "source_csv_or_tex": "generated/summary_by_method.csv",
                "generation_script": "scripts/aggregate_results.py",
                "reproduction_command": "python scripts/aggregate_results.py",
                "status": "todo",
                "notes": "",
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "phase": "main",
                "tier": "small",
                "method_id": "EDE",
                "summary_scope": "accepted_only",
                "source_runs": 10,
                "accepted_runs": 10,
                "strict_duty_runs": 10,
                "accepted_rate": 100.0,
                "strict_duty_rate": 100.0,
                "runs": 10,
                "median_j": 1.0,
                "median_cost": 1.0,
                "median_energy": 1.0,
                "median_co2": 1.0,
                "median_runtime": 1.0,
                "median_wh": 1.0,
                "median_imp_per_wh": 1.0,
            }
        ]
    )

    completed = template.apply(module._fill_claim_row, axis=1, summary=summary, stats=None)
    assert completed.iloc[0]["status"] == "supported_by_artifacts"
