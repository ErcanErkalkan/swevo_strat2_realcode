from __future__ import annotations

from .base import BaseComparator
from ..benchmark import build_problem, load_scenarios
from ..solver import MetaheuristicConfig, default_population_size, jde_evolve


class AblationComparator(BaseComparator):
    def __init__(self, method_id: str = "A1_NoSeed"):
        super().__init__(method_id=method_id)

    def config_for_plan(self, plan):
        pop = default_population_size(plan.customer_count)
        if self.method_id == "A1_NoSeed":
            return MetaheuristicConfig(pop, plan.eval_budget, plan.seed, use_seed=False, use_jde=True, use_lns=True, lns_period=10, repair_budget=14, local_search_moves=24)
        if self.method_id == "A2_NoJDE":
            return MetaheuristicConfig(pop, plan.eval_budget, plan.seed, use_seed=True, use_jde=False, use_lns=True, lns_period=10, repair_budget=14, local_search_moves=24, fixed_F=0.72, fixed_CR=0.88)
        if self.method_id == "A3_NoLNS":
            return MetaheuristicConfig(pop, plan.eval_budget, plan.seed, use_seed=True, use_jde=True, use_lns=False, lns_period=10, repair_budget=14, local_search_moves=24)
        return MetaheuristicConfig(pop, plan.eval_budget, plan.seed)

    def solve(self, plan):
        problem = build_problem(plan, load_scenarios())
        cfg = self.config_for_plan(plan)
        best, stats, archive = jde_evolve(problem, cfg, source_tag=self.method_id)
        stats.note(f"problem_source={problem.source_kind}")
        if problem.source_path:
            stats.note(f"problem_path={problem.source_path}")
        return best, stats, archive
