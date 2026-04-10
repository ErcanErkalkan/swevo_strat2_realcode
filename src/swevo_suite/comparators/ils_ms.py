from __future__ import annotations

from .base import BaseComparator
from ..benchmark import build_problem, load_scenarios
from ..solver import MetaheuristicConfig, ils_search
from ..schemas import RunPlan


class ILSMSComparator(BaseComparator):
    def __init__(self):
        super().__init__(method_id="ILS_MS")

    def config_for_plan(self, plan: RunPlan) -> MetaheuristicConfig:
        return MetaheuristicConfig(
            population_size=1,
            eval_budget=plan.eval_budget,
            seed=plan.seed,
            walltime_cap_s=plan.walltime_cap_s,
            repair_budget=14,
            local_search_moves=28 if plan.customer_count <= 120 else 22 if plan.customer_count <= 240 else 18,
        )

    def solve(self, plan: RunPlan):
        problem = build_problem(plan, load_scenarios())
        cfg = self.config_for_plan(plan)
        best, stats, archive = ils_search(problem, cfg)
        stats.note(f"problem_source={problem.source_kind}")
        if problem.source_path:
            stats.note(f"problem_path={problem.source_path}")
        return best, stats, archive
