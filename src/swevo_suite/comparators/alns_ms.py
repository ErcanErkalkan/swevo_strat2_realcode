from __future__ import annotations

from .base import BaseComparator
from ..benchmark import build_problem, load_scenarios
from ..solver import MetaheuristicConfig, alns_search
from ..schemas import RunPlan


class ALNSMSComparator(BaseComparator):
    def __init__(self):
        super().__init__(method_id="ALNS_MS")

    def config_for_plan(self, plan: RunPlan) -> MetaheuristicConfig:
        return MetaheuristicConfig(
            population_size=1,
            eval_budget=plan.eval_budget,
            seed=plan.seed,
            repair_budget=16,
            local_search_moves=26 if plan.customer_count <= 120 else 22 if plan.customer_count <= 240 else 18,
        )

    def solve(self, plan: RunPlan):
        problem = build_problem(plan, load_scenarios())
        cfg = self.config_for_plan(plan)
        best, stats, archive = alns_search(problem, cfg)
        stats.note(f"problem_source={problem.source_kind}")
        if problem.source_path:
            stats.note(f"problem_path={problem.source_path}")
        return best, stats, archive
