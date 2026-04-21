from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swevo_suite.stats import pairwise_wilcoxon, summary_by_method


def sample_runs() -> pd.DataFrame:
    rows = [
        ("run-ede-1", "main", "small", "EDE", "C101", "S1", 1, 1, 1, 1.0),
        ("run-std-1", "main", "small", "StdDE", "C101", "S1", 1, 0, 0, 2.0),
        ("run-ils-1", "main", "small", "ILS_MS", "C101", "S1", 1, 1, 1, 0.9),
        ("run-ede-2", "main", "small", "EDE", "C101", "S1", 2, 1, 1, 0.8),
        ("run-std-2", "main", "small", "StdDE", "C101", "S1", 2, 1, 1, 1.5),
        ("run-ils-2", "main", "small", "ILS_MS", "C101", "S1", 2, 1, 1, 0.85),
    ]
    data = []
    for run_id, phase, tier, method_id, instance_id, scenario_id, seed, accepted, strict, score in rows:
        data.append(
            {
                "run_id": run_id,
                "phase": phase,
                "tier": tier,
                "method_id": method_id,
                "instance_id": instance_id,
                "scenario_id": scenario_id,
                "seed": seed,
                "accepted_final": accepted,
                "strict_duty_final": strict,
                "j_scaled_final": score,
                "cost_final": score * 100.0,
                "energy_final": score * 10.0,
                "co2_final": score,
                "runtime_sec": score,
                "compute_wh": score / 10.0,
                "imp_per_wh": score / 100.0,
            }
        )
    return pd.DataFrame(data)


def test_summary_by_method_tracks_accepted_only_view() -> None:
    df = sample_runs()

    overall = summary_by_method(df)
    accepted = summary_by_method(df, accepted_only=True)

    std_overall = overall[overall["method_id"] == "StdDE"].iloc[0]
    std_accepted = accepted[accepted["method_id"] == "StdDE"].iloc[0]

    assert std_overall["summary_scope"] == "unconditional"
    assert std_overall["source_runs"] == 2
    assert std_overall["runs"] == 2
    assert std_overall["accepted_rate"] == 50.0
    assert std_overall["median_j"] == 1.75

    assert std_accepted["summary_scope"] == "accepted_only"
    assert std_accepted["source_runs"] == 2
    assert std_accepted["runs"] == 1
    assert std_accepted["accepted_rate"] == 50.0
    assert std_accepted["median_j"] == 1.5


def test_pairwise_wilcoxon_can_filter_to_accepted_only() -> None:
    df = sample_runs()

    overall = pairwise_wilcoxon(df, control_method="EDE")
    accepted = pairwise_wilcoxon(df, control_method="EDE", accepted_only=True)

    std_overall = overall[overall["compare_method"] == "StdDE"].iloc[0]
    std_accepted = accepted[accepted["compare_method"] == "StdDE"].iloc[0]

    assert std_overall["summary_scope"] == "unconditional"
    assert std_overall["n_pairs"] == 2
    assert std_accepted["summary_scope"] == "accepted_only"
    assert std_accepted["n_pairs"] == 1
