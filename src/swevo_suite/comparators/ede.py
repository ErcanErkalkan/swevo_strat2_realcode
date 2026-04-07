from __future__ import annotations

from .base import BaseComparator
from ..benchmark import build_problem, load_scenarios
from ..solver import MetaheuristicConfig, default_population_size, jde_evolve
from ..schemas import RunPlan


class EDEComparator(BaseComparator):
    def __init__(self):
        super().__init__(method_id="EDE")

    def config_for_plan(self, plan: RunPlan) -> MetaheuristicConfig:
        pop = default_population_size(plan.customer_count)
        return MetaheuristicConfig(
            population_size=pop,
            eval_budget=plan.eval_budget,
            seed=plan.seed,
            use_seed=True,
            use_jde=True,
            use_lns=True,
            lns_period=10,
            repair_budget=14,
            local_search_moves=28 if plan.customer_count <= 120 else 24 if plan.customer_count <= 240 else 20,
            diversity_restart=True,
            fixed_F=0.72,
            fixed_CR=0.88,
        )

    def solve(self, plan: RunPlan):
        problem = build_problem(plan, load_scenarios())
        cfg = self.config_for_plan(plan)
        best, stats, archive = jde_evolve(problem, cfg, source_tag=self.method_id)
        stats.note(f"problem_source={problem.source_kind}")
        if problem.source_path:
            stats.note(f"problem_path={problem.source_path}")
        return best, stats, archive
