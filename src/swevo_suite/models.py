from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass(frozen=True)
class Customer:
    idx: int
    x: float
    y: float
    demand: float
    tw_start: float
    tw_end: float
    service: float
    arc_zone: str


@dataclass(frozen=True)
class Shift:
    idx: int
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VehicleType:
    name: str
    capacity: float
    fixed_cost: float
    energy_alpha: float
    energy_beta: float
    emission_factor: float


@dataclass
class RouteMetrics:
    cost: float = 0.0
    energy: float = 0.0
    co2: float = 0.0
    overtime: float = 0.0
    overtime_ratio: float = 0.0
    v_cap: float = 0.0
    v_tw: float = 0.0
    v_shift: float = 0.0
    return_time: float = 0.0
    service_times: Dict[int, float] = field(default_factory=dict)


@dataclass
class Route:
    shift_id: int
    vehicle_type: str
    customers: List[int] = field(default_factory=list)
    metrics: RouteMetrics = field(default_factory=RouteMetrics)


@dataclass
class Solution:
    routes: List[Route]
    references: Tuple[float, float, float]
    source: str = ""
    score: float = float("inf")
    cost: float = float("inf")
    energy: float = float("inf")
    co2: float = float("inf")
    overtime_sum: float = 0.0
    overtime_ratio_sum: float = 0.0
    v_cap: float = 0.0
    v_tw: float = 0.0
    v_shift: float = 0.0
    accepted: bool = False
    strict_duty: bool = False

    def copy(self) -> "Solution":
        return Solution(
            routes=[Route(r.shift_id, r.vehicle_type, list(r.customers), RouteMetrics(**r.metrics.__dict__)) for r in self.routes],
            references=self.references,
            source=self.source,
            score=self.score,
            cost=self.cost,
            energy=self.energy,
            co2=self.co2,
            overtime_sum=self.overtime_sum,
            overtime_ratio_sum=self.overtime_ratio_sum,
            v_cap=self.v_cap,
            v_tw=self.v_tw,
            v_shift=self.v_shift,
            accepted=self.accepted,
            strict_duty=self.strict_duty,
        )


@dataclass
class ProblemInstance:
    name: str
    family: str
    structure_class: str
    customer_count: int
    customers: Dict[int, Customer]
    shifts: List[Shift]
    depot_xy: Tuple[float, float]
    vehicle_types: Dict[str, VehicleType]
    shift_factors: Dict[int, float]
    zone_factors: Dict[str, float]
    source_kind: str = "synthetic"
    source_path: str = ""
    eta: float = 0.05
    epsilon: float = 1e-9
    objective_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    overtime_penalty: float = 0.25

    @property
    def customer_ids(self) -> List[int]:
        return list(self.customers.keys())


@dataclass
class SearchStats:
    eval_count: int = 0
    n_repair_attempts: int = 0
    n_repair_success: int = 0
    n_rejected_offspring: int = 0
    archive_size_final: int = 0
    first_feasible_eval: Optional[int] = None
    first_feasible_sec: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.notes.append(msg)
