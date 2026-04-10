from __future__ import annotations

from .base import BaseComparator
from ..benchmark import build_problem, load_scenarios
from ..solver import MetaheuristicConfig, default_population_size, hgs_search
from ..schemas import RunPlan


class HGSMSComparator(BaseComparator):
    def __init__(self):
        super().__init__(method_id="HGS_MS")

    def config_for_plan(self, plan: RunPlan) -> MetaheuristicConfig:
        pop = max(18, default_population_size(plan.customer_count) - 6)
        return MetaheuristicConfig(
            population_size=pop,
            eval_budget=plan.eval_budget,
            seed=plan.seed,
            walltime_cap_s=plan.walltime_cap_s,
            repair_budget=14,
            local_search_moves=24 if plan.customer_count <= 120 else 20 if plan.customer_count <= 240 else 16,
        )

    def solve(self, plan: RunPlan):
        problem = build_problem(plan, load_scenarios())
        cfg = self.config_for_plan(plan)
        best, stats, archive = hgs_search(problem, cfg)
        stats.note(f"problem_source={problem.source_kind}")
        if problem.source_path:
            stats.note(f"problem_path={problem.source_path}")
        return best, stats, archive
