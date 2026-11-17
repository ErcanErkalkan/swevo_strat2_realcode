from __future__ import annotations

from .base import BaseComparator
from .ede import EDEComparator
from ..benchmark import build_problem, load_scenarios
from ..solver import MetaheuristicConfig, jde_evolve


class AblationComparator(BaseComparator):
    """Single-concept ablations derived from the canonical EDE configuration.

    This deliberately clones EDE's current configuration first. That prevents
    silent drift when EDE gains new intensification or endgame components.
    """

    def __init__(self, method_id: str = "A1_NoSeed"):
        super().__init__(method_id=method_id)

    def config_for_plan(self, plan) -> MetaheuristicConfig:
        cfg = EDEComparator().config_for_plan(plan)

        if self.method_id == "A1_NoSeed":
            # Remove only the informed initialization/seed branch.
            cfg.use_seed = False
            return cfg

        if self.method_id == "A2_NoJDE":
            # Remove only jDE self-adaptation of F/CR. Keep EDE's mutation
            # strategy, LNS, trajectory search, deep intensification, and
            # route-ALNS endgame unchanged.
            cfg.use_jde = False
            cfg.use_current_to_best = True
            return cfg

        if self.method_id == "A3_NoLNS":
            # Treat periodic boundary-LNS and route-ALNS endgame as one
            # conceptual LNS family; keep all non-LNS EDE components intact.
            cfg.use_lns = False
            cfg.use_route_alns_endgame = False
            return cfg

        raise ValueError(f"Unknown ablation method_id: {self.method_id}")

    def solve(self, plan):
        problem = build_problem(plan, load_scenarios())
        cfg = self.config_for_plan(plan)
        best, stats, archive = jde_evolve(problem, cfg, source_tag=self.method_id)
        stats.note(f"problem_source={problem.source_kind}")
        if problem.source_path:
            stats.note(f"problem_path={problem.source_path}")
        return best, stats, archive
