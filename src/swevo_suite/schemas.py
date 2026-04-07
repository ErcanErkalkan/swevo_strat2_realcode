from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class RunPlan:
    run_id: str
    phase: str
    benchmark_family: str
    customer_count: int
    instance_id: str
    structure_class: str
    scenario_id: str
    method_id: str
    method_group: str
    ablation_flag: str
    seed: int
    tier: str
    eval_budget: int
    walltime_cap_s: int
    status: str

@dataclass
class RunResult:
    run_id: str
    phase: str
    benchmark_family: str
    customer_count: int
    instance_id: str
    structure_class: str
    scenario_id: str
    method_id: str
    seed: int
    tier: str
    eval_budget: int
    accepted_final: int
    strict_duty_final: int
    v_cap_final: float
    v_tw_final: float
    v_shift_final: float
    overtime_sum_final: float
    overtime_ratio_sum_final: float
    cost_final: float
    energy_final: float
    co2_final: float
    j_scaled_init: float
    j_scaled_final: float
    improvement_abs: float
    improvement_pct: float
    runtime_sec: float
    compute_wh: float
    imp_per_wh: float
    time_to_first_feasible_sec: float
    evals_to_first_feasible: int
    n_repair_attempts: int
    n_repair_success: int
    n_rejected_offspring: int
    archive_size_final: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
