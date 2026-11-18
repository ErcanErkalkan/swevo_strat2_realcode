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


def _primary_summary_fixture() -> pd.DataFrame:
    methods = ["EDE", "StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"]
    return pd.DataFrame(
        [
            {
                "phase": "main",
                "tier": "all",
                "method_id": method,
                "summary_scope": "accepted_only",
                "source_runs": 2160,
                "accepted_runs": 2160,
                "strict_duty_runs": 2160,
                "accepted_rate": 100.0,
                "strict_duty_rate": 100.0,
                "runs": 2160,
                "median_j": 0.75 if method == "EDE" else 1.0,
                "median_cost": 1.0,
                "median_energy": 1.0,
                "median_co2": 1.0,
                "median_runtime": 60.0,
                "median_wh": 1.43,
                "median_imp_per_wh": 0.056 if method == "EDE" else 0.015,
            }
            for method in methods
        ]
    )


def _primary_stats_fixture() -> pd.DataFrame:
    comparisons = ["StdDE", "ALNS_MS", "HGS_MS", "ILS_MS"]
    pairwise = [
        {
            "metric": "j_scaled_final",
            "control_method": "EDE",
            "compare_method": method,
            "n_pairs": 108,
            "wilcoxon_stat": 0.0,
            "p_value": 1e-12,
            "median_diff_control_minus_compare": -0.1,
            "rank_biserial": 1.0,
            "p_holm": 4e-12,
            "summary_scope": "accepted_only",
            "section": "pairwise",
            "blocks": None,
            "k_methods": None,
            "friedman_chi2": None,
            "kendalls_w": None,
        }
        for method in comparisons
    ]
    omnibus = {
        "metric": "j_scaled_final",
        "control_method": None,
        "compare_method": None,
        "n_pairs": None,
        "wilcoxon_stat": None,
        "p_value": 1e-20,
        "median_diff_control_minus_compare": None,
        "rank_biserial": None,
        "p_holm": None,
        "summary_scope": "accepted_only",
        "section": "omnibus",
        "blocks": 108,
        "k_methods": 5,
        "friedman_chi2": 324.5,
        "kendalls_w": 0.75,
    }
    return pd.DataFrame(pairwise + [omnibus])


def test_claim_evidence_map_marks_supported_claims(tmp_path: Path) -> None:
    module = _load_script_module()

    template = pd.DataFrame(
        [
            {"claim_id": "C01", "status": "todo", "notes": ""},
            {"claim_id": "C02", "status": "todo", "notes": ""},
            {"claim_id": "C03", "status": "todo", "notes": ""},
            {"claim_id": "C04", "status": "todo", "notes": ""},
        ]
    )
    completed = template.apply(
        module._fill_claim_row,
        axis=1,
        summary=_primary_summary_fixture(),
        stats=_primary_stats_fixture(),
    )

    statuses = completed.set_index("claim_id")["status"].to_dict()
    assert statuses == {
        "C01": "supported_by_artifacts",
        "C02": "supported_by_artifacts",
        "C03": "supported_by_artifacts",
        "C04": "supported_by_code_and_artifacts",
    }


def test_claim_evidence_map_normalizes_claim_id_case_and_whitespace() -> None:
    module = _load_script_module()
    template = pd.DataFrame([{"claim_id": " c01 ", "status": "todo", "notes": ""}])
    completed = template.apply(
        module._fill_claim_row,
        axis=1,
        summary=_primary_summary_fixture(),
        stats=None,
    )
    assert completed.iloc[0]["status"] == "supported_by_artifacts"



def test_c12_secondary_operational_semantics() -> None:
    module = _load_script_module()
    template = pd.DataFrame([{"claim_id":"C12","status":"todo","notes":""}])
    operational = pd.DataFrame([{
        "scope":"all","runs":18,"hard_feasible_runs":18,"strict_duty_runs":15,
        "bounded_overtime_runs":3,"max_overtime_sum":0.341028,
        "max_v_cap":0.0,"max_v_tw":0.0,"max_v_shift":0.0,
    }])
    completed = template.apply(module._fill_claim_row, axis=1, summary=None, stats=None, operational=operational)
    assert completed.iloc[0]["status"] == "supported_as_secondary_semantics"


def test_c13_secondary_parameter_sensitivity() -> None:
    module = _load_script_module()
    template = pd.DataFrame([{"claim_id":"C13","status":"todo","notes":""}])
    sensitivity = pd.DataFrame([
        {"config":"BASE","runs":9,"hard_feasible_runs":9,"max_abs_pct_delta_vs_base":0.0},
        {"config":"LNS5","runs":9,"hard_feasible_runs":9,"max_abs_pct_delta_vs_base":1.98},
        {"config":"LNS20","runs":9,"hard_feasible_runs":9,"max_abs_pct_delta_vs_base":1.56},
        {"config":"REPAIR8","runs":9,"hard_feasible_runs":9,"max_abs_pct_delta_vs_base":1.56},
        {"config":"REPAIR20","runs":9,"hard_feasible_runs":9,"max_abs_pct_delta_vs_base":2.30},
    ])
    stats = pd.DataFrame([
        {"section":"pairwise_vs_BASE","compare_config":"LNS5","p_holm":0.75},
        {"section":"pairwise_vs_BASE","compare_config":"LNS20","p_holm":0.75},
        {"section":"pairwise_vs_BASE","compare_config":"REPAIR8","p_holm":0.75},
        {"section":"pairwise_vs_BASE","compare_config":"REPAIR20","p_holm":0.25},
    ])
    completed = template.apply(module._fill_claim_row, axis=1, summary=None, stats=None, parameter_sensitivity=sensitivity, parameter_sensitivity_stats=stats)
    assert completed.iloc[0]["status"] == "supported_as_secondary_local_sensitivity"


def test_c14_secondary_anytime_profile() -> None:
    module = _load_script_module()
    template = pd.DataFrame([{"claim_id":"C14","status":"todo","notes":""}])
    summary = pd.DataFrame([
        {"instance_id":inst,"walltime_cap_s":cap,"hard_feasible_runs":4,"strict_duty_runs":4,"ede_median_better":1}
        for inst in ["C101","R101","RC101"] for cap in [1,2,4,8]
    ])
    auc = pd.DataFrame([
        {"instance_id":inst,"seed":seed,"ede_better_auc":1}
        for inst in ["C101","R101","RC101"] for seed in [1,2]
    ])
    overall = pd.DataFrame([{
        "runs":48,"hard_feasible_runs":48,"strict_duty_runs":48,
        "instance_cap_cells":12,"ede_median_better_cells":12,
        "paired_seed_cap_comparisons":24,"ede_seed_wins":23,
        "auc_trajectories":6,"ede_better_auc_trajectories":6,
        "median_auc_pct_reduction_ede_vs_stdde":9.267,
        "max_runtime_over_cap_s":0.05,
    }])
    completed = template.apply(
        module._fill_claim_row, axis=1, summary=None, stats=None,
        anytime_summary=summary, anytime_auc=auc, anytime_overall=overall,
    )
    assert completed.iloc[0]["status"] == "supported_as_secondary_anytime"


def test_current_map_has_no_missing_or_stale_reproduction_paths():
    current = pd.read_csv(ROOT / "results/primary/claim_evidence_map.csv")
    assert len(current) == 14
    assert current.status.map(lambda x: x.startswith("supported_") or x == "scope_declared").all()
    assert current.reproduction_command.eq("python scripts/finalize.py --core").all()
    for field in ["source_csv_or_tex", "generation_script"]:
        assert not current[field].str.contains("generated_submission|finalize_submission_state|build_submission_strengthening_evidence").any()
    assert "Instance-cluster" in current.set_index("claim_id").loc["C09", "exact_claim_text"]
