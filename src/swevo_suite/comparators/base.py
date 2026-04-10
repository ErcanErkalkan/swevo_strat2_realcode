from __future__ import annotations

import time
from dataclasses import dataclass

from ..schemas import RunPlan, RunResult
from ..solver import MetaheuristicConfig, alns_search, default_population_size, hgs_search, ils_search, jde_evolve


@dataclass
class BaseComparator:
    method_id: str

    def config_for_plan(self, plan: RunPlan) -> MetaheuristicConfig:
        return MetaheuristicConfig(
            population_size=default_population_size(plan.customer_count),
            eval_budget=plan.eval_budget,
            seed=plan.seed,
            walltime_cap_s=plan.walltime_cap_s,
        )

    def solve(self, plan: RunPlan):
        raise NotImplementedError

    def _power_watts(self, tier: str) -> float:
        return {"small": 52.0, "medium": 86.0, "large": 128.0}.get(tier, 75.0)

    def run(self, plan: RunPlan) -> RunResult:
        t0 = time.perf_counter()
        best, stats, archive = self.solve(plan)
        runtime = time.perf_counter() - t0
        compute_wh = runtime * self._power_watts(plan.tier) / 3600.0
        init_best = None
        for note in stats.notes:
            if note.startswith("init_best="):
                init_best = float(note.split("=", 1)[1])
        if init_best is None:
            init_best = best.score
        improvement_abs = max(0.0, init_best - best.score)
        improvement_pct = 100.0 * improvement_abs / max(init_best, 1e-9)
        notes = [f"real_solver:{self.method_id}", f"archive={len(archive)}", *stats.notes]
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
            accepted_final=int(best.accepted),
            strict_duty_final=int(best.strict_duty),
            v_cap_final=float(best.v_cap),
            v_tw_final=float(best.v_tw),
            v_shift_final=float(best.v_shift),
            overtime_sum_final=float(best.overtime_sum),
            overtime_ratio_sum_final=float(best.overtime_ratio_sum),
            cost_final=float(best.cost),
            energy_final=float(best.energy),
            co2_final=float(best.co2),
            j_scaled_init=float(init_best),
            j_scaled_final=float(best.score),
            improvement_abs=float(improvement_abs),
            improvement_pct=float(improvement_pct),
            runtime_sec=float(runtime),
            compute_wh=float(compute_wh),
            imp_per_wh=float(improvement_abs / max(compute_wh, 1e-9)),
            time_to_first_feasible_sec=float(stats.first_feasible_sec or runtime),
            evals_to_first_feasible=int(stats.first_feasible_eval or stats.eval_count),
            n_repair_attempts=int(stats.n_repair_attempts),
            n_repair_success=int(stats.n_repair_success),
            n_rejected_offspring=int(stats.n_rejected_offspring),
            archive_size_final=int(stats.archive_size_final),
            notes=" | ".join(notes),
        )


class ComparatorNotImplementedError(NotImplementedError):
    pass
