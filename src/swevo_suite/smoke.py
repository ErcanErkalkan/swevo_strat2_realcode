from __future__ import annotations
import random
from .schemas import RunPlan, RunResult

METHOD_OFFSET = {
    "EDE": -0.12,
    "StdDE": 0.04,
    "ALNS_MS": -0.05,
    "HGS_MS": -0.06,
    "ILS_MS": -0.02,
    "A1_NoSeed": -0.01,
    "A2_NoJDE": 0.00,
    "A3_NoLNS": 0.01,
}

TIER_SCALE = {"small": 1.0, "medium": 3.0, "large": 7.0}
STRUCTURE_SCALE = {"clustered": 0.95, "mixed": 1.00, "random": 1.05}

def synthesize_result(plan: RunPlan) -> RunResult:
    rng = random.Random(hash((plan.run_id, plan.seed, plan.method_id)) & 0xffffffff)
    base = 200.0 * TIER_SCALE[plan.tier] * STRUCTURE_SCALE[plan.structure_class]
    scenario_shift = {"S1_balanced": 1.0, "S2_peak_dirty": 1.06, "S3_mixed_fleet_arc": 1.11}[plan.scenario_id]
    method_shift = METHOD_OFFSET.get(plan.method_id, 0.0)
    cost = base * scenario_shift * (1.0 + method_shift + rng.uniform(-0.02, 0.02))
    energy = base * 0.065 * scenario_shift * (1.0 + method_shift + rng.uniform(-0.025, 0.025))
    co2 = energy * {"S1_balanced": 2.25, "S2_peak_dirty": 2.45, "S3_mixed_fleet_arc": 2.10}[plan.scenario_id]
    j_init = base * 1.30 * (1.0 + rng.uniform(-0.03, 0.03))
    j_final = base * (1.0 + method_shift + rng.uniform(-0.03, 0.03))
    improvement_abs = max(0.0, j_init - j_final)
    improvement_pct = 100.0 * improvement_abs / max(j_init, 1e-9)
    runtime = {"small": 120.0, "medium": 480.0, "large": 1800.0}[plan.tier] * (1.0 + rng.uniform(-0.15, 0.15))
    compute_wh = runtime / 3600.0 * {"small": 45, "medium": 85, "large": 140}[plan.tier] * (1.0 + rng.uniform(-0.12, 0.12))
    accepted = 1
    overtime = 0.0 if rng.random() < 0.93 else round(rng.uniform(0.1, 4.0), 4)
    strict = 1 if overtime == 0 else 0
    if plan.method_id == "StdDE" and plan.tier == "large" and rng.random() < 0.08:
        accepted = 0
    if accepted == 0:
        vtw = round(rng.uniform(0.1, 5.0), 4)
        vcap = 0.0
        vshift = 0.0
        strict = 0
    else:
        vtw = vcap = vshift = 0.0
    tff = runtime * rng.uniform(0.08, 0.35)
    eff = int(plan.eval_budget * rng.uniform(0.07, 0.32))
    return RunResult(
        run_id=plan.run_id,
        phase=plan.phase,
        benchmark_family=plan.benchmark_family,
        customer_count=plan.customer_count,
        instance_id=plan.instance_id,
        structure_class=plan.structure_class,
        scenario_id=plan.scenario_id,
        method_id=plan.method_id,
        seed=plan.seed,
        tier=plan.tier,
        eval_budget=plan.eval_budget,
        accepted_final=accepted,
        strict_duty_final=strict,
        v_cap_final=vcap,
        v_tw_final=vtw,
        v_shift_final=vshift,
        overtime_sum_final=overtime,
        overtime_ratio_sum_final=0.0 if overtime == 0 else min(1.0, overtime / 12.0),
        cost_final=round(cost, 6),
        energy_final=round(energy, 6),
        co2_final=round(co2, 6),
        j_scaled_init=round(j_init, 6),
        j_scaled_final=round(j_final, 6),
        improvement_abs=round(improvement_abs, 6),
        improvement_pct=round(improvement_pct, 6),
        runtime_sec=round(runtime, 6),
        compute_wh=round(compute_wh, 6),
        imp_per_wh=round(improvement_abs / max(compute_wh, 1e-9), 6),
        time_to_first_feasible_sec=round(tff, 6),
        evals_to_first_feasible=eff,
        n_repair_attempts=int(plan.eval_budget * rng.uniform(0.01, 0.05)),
        n_repair_success=int(plan.eval_budget * rng.uniform(0.008, 0.03)),
        n_rejected_offspring=int(plan.eval_budget * rng.uniform(0.02, 0.09)),
        archive_size_final=int(rng.uniform(8, 48)),
        notes="synthetic_smoke_only"
    )
