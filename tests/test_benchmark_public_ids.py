from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swevo_suite.benchmark import InvalidBenchmarkInstanceId, build_problem, public_instance_id_error
from swevo_suite.schemas import RunPlan


def _plan(instance_id: str) -> RunPlan:
    return RunPlan(
        run_id=f"{instance_id}__S1__EDE__seed01",
        phase="main",
        benchmark_family="solomon",
        customer_count=100,
        instance_id=instance_id,
        structure_class="mixed",
        scenario_id="S1_balanced",
        method_id="EDE",
        method_group="main",
        ablation_flag="",
        seed=1,
        tier="small",
        eval_budget=1000,
        walltime_cap_s=60,
        status="planned",
    )


def test_public_instance_id_error_distinguishes_valid_and_invalid_solomon_ids() -> None:
    assert public_instance_id_error(_plan("RC108")) is None

    message = public_instance_id_error(_plan("RC109"))
    assert message is not None
    assert "public Solomon-100 benchmark set" in message
    assert "RC101-RC108" in message


def test_build_problem_raises_invalid_public_instance_when_real_benchmarks_are_required() -> None:
    with pytest.raises(InvalidBenchmarkInstanceId, match="RC109"):
        build_problem(_plan("RC109"), require_real=True)
