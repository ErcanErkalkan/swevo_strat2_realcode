from __future__ import annotations

from .base import BaseComparator
from ..benchmark import build_problem, load_scenarios
from ..solver import MetaheuristicConfig, default_population_size, jde_evolve
from ..schemas import RunPlan


class StdDEComparator(BaseComparator):
    def __init__(self):
        super().__init__(method_id="StdDE")

    def config_for_plan(self, plan: RunPlan) -> MetaheuristicConfig:
        pop = default_population_size(plan.customer_count)
        return MetaheuristicConfig(
            population_size=pop,
            eval_budget=plan.eval_budget,
            seed=plan.seed,
            walltime_cap_s=plan.walltime_cap_s,
            use_seed=False,
            use_jde=False,
            use_lns=False,
            repair_budget=10,
            local_search_moves=12 if plan.customer_count <= 120 else 10,
            diversity_restart=False,
            fixed_F=0.75,
            fixed_CR=0.90,
        )

    def solve(self, plan: RunPlan):
        problem = build_problem(plan, load_scenarios())
        cfg = self.config_for_plan(plan)
        best, stats, archive = jde_evolve(problem, cfg, source_tag=self.method_id)
        stats.note(f"problem_source={problem.source_kind}")
        if problem.source_path:
            stats.note(f"problem_path={problem.source_path}")
        return best, stats, archive
