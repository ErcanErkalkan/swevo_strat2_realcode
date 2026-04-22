from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swevo_suite.comparators.ede import EDEComparator
from swevo_suite.schemas import RunPlan


def _plan() -> RunPlan:
    return RunPlan(
        run_id="C101__S1_balanced__EDE__seed01",
        phase="main",
        benchmark_family="solomon",
        customer_count=100,
        instance_id="C101",
        structure_class="clustered",
        scenario_id="S1_balanced",
        method_id="EDE",
        method_group="proposed",
        ablation_flag="full",
        seed=1,
        tier="small",
        eval_budget=25050,
        walltime_cap_s=60,
        status="planned",
    )


def test_ede_config_reads_route_endgame_burst_iters_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SWEVO_EDE_ROUTE_ENDGAME_BURST_ITERS", "6")
    cfg = EDEComparator().config_for_plan(_plan())
    assert cfg.route_endgame_burst_iters == 6
