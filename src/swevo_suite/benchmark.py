from __future__ import annotations

import ast
import csv
import hashlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .models import Customer, ProblemInstance, Shift, VehicleType
from .paths import CONFIGS, ROOT
from .schemas import RunPlan


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    vehicle_factors: Dict[str, float]
    shift_factors: List[float]
    zone_factors: Dict[str, float]


BENCHMARK_ROOT = ROOT / "data" / "benchmarks"
INSTANCE_SUFFIXES = {".txt", ".vrp", ".csv"}


class InvalidBenchmarkInstanceId(FileNotFoundError):
    """Raised when a manifest row names an instance that is not in the public set."""


def _stable_seed(*parts: object) -> int:
    h = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFF


def _parse_factor_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in text.split(","):
        key, value = item.strip().split(":")
        out[key.strip()] = float(value)
    return out


def load_scenarios(path: Path | None = None) -> Dict[str, ScenarioSpec]:
    scenario_path = path or (CONFIGS / "scenario_registry.csv")
    scenarios: Dict[str, ScenarioSpec] = {}
    with scenario_path.open() as f:
        for row in csv.DictReader(f):
            scenarios[row["scenario_id"]] = ScenarioSpec(
                scenario_id=row["scenario_id"],
                vehicle_factors=_parse_factor_map(row["psi_vehicle"]),
                shift_factors=[float(x) for x in ast.literal_eval(row["psi_shift"])],
                zone_factors=_parse_factor_map(row["psi_arc"]),
            )
    return scenarios


def _sample_cluster_centers(rng: np.random.Generator, n_clusters: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    radii = rng.uniform(18, 42, size=n_clusters)
    centers = np.stack([50 + radii * np.cos(angles), 50 + radii * np.sin(angles)], axis=1)
    centers += rng.normal(0, 4, size=centers.shape)
    return centers


def _make_coords(
    rng: np.random.Generator,
    n: int,
    structure_class: str,
) -> np.ndarray:
    if structure_class == "clustered":
        k = 6 if n <= 120 else 8 if n <= 240 else 10
        centers = _sample_cluster_centers(rng, k)
        labels = rng.integers(0, k, size=n)
        coords = centers[labels] + rng.normal(0, 4.5, size=(n, 2))
    elif structure_class == "random":
        coords = rng.uniform(0, 100, size=(n, 2))
    else:
        k = 5 if n <= 120 else 7 if n <= 240 else 9
        centers = _sample_cluster_centers(rng, k)
        labels = rng.integers(0, k, size=n)
        cluster_coords = centers[labels] + rng.normal(0, 5.5, size=(n, 2))
        random_coords = rng.uniform(0, 100, size=(n, 2))
        mask = rng.random(n) < 0.6
        coords = np.where(mask[:, None], cluster_coords, random_coords)
    return np.clip(coords, 0, 100)


def _arc_zone(x: float, y: float, depot_xy: Tuple[float, float]) -> str:
    dx = x - depot_xy[0]
    dy = y - depot_xy[1]
    r = math.hypot(dx, dy)
    if r < 18:
        return "urban_core"
    if r < 36:
        return "ring"
    return "suburban"


def _tw_profile(rng: np.random.Generator, idx: int, x: float, y: float, tier: str) -> Tuple[float, float, float]:
    horizon = {"small": 960.0, "medium": 1260.0, "large": 1620.0}[tier]
    bucket = idx % 3
    base_centers = [0.18 * horizon, 0.50 * horizon, 0.79 * horizon]
    center = base_centers[bucket] + rng.normal(0, 0.05 * horizon)
    width = rng.uniform(0.10, 0.16) * horizon if tier == "small" else rng.uniform(0.08, 0.13) * horizon
    start = max(0.0, center - width / 2.0)
    end = min(horizon, center + width / 2.0)
    service = rng.uniform(8, 18) if tier == "small" else rng.uniform(7, 16)
    return start, max(start + service + 15.0, end), service


def _instance_family_dir(family: str) -> str:
    fam = family.lower()
    if "solomon" in fam:
        return "solomon"
    if "homberger" in fam or "hg" in fam:
        return "homberger"
    if "li" in fam:
        return "li_lim"
    return fam


def _solomon_public_instance_ids() -> frozenset[str]:
    instance_ids: set[str] = set()
    instance_ids.update(f"C{idx}" for idx in range(101, 110))
    instance_ids.update(f"C{idx}" for idx in range(201, 209))
    instance_ids.update(f"R{idx}" for idx in range(101, 113))
    instance_ids.update(f"R{idx}" for idx in range(201, 212))
    instance_ids.update(f"RC{idx}" for idx in range(101, 109))
    instance_ids.update(f"RC{idx}" for idx in range(201, 209))
    return frozenset(instance_ids)


KNOWN_PUBLIC_INSTANCE_IDS = {
    "solomon": _solomon_public_instance_ids(),
}


def public_instance_id_error(plan: RunPlan) -> str | None:
    family_key = _instance_family_dir(plan.benchmark_family)
    instance_id = plan.instance_id.upper()
    if family_key == "solomon" and plan.customer_count == 100:
        valid_ids = KNOWN_PUBLIC_INSTANCE_IDS["solomon"]
        if instance_id not in valid_ids:
            return (
                f"{plan.instance_id} is not part of the public Solomon-100 benchmark set. "
                "Valid mixed-instance ids are RC101-RC108 and RC201-RC208."
                if instance_id.startswith("RC")
                else f"{plan.instance_id} is not part of the public Solomon-100 benchmark set."
            )
    return None


@lru_cache(maxsize=None)
def _benchmark_file_index(root_key: str) -> Dict[str, Tuple[Path, ...]]:
    root = Path(root_key)
    index: Dict[str, list[Path]] = {}
    if not root.exists():
        return {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in INSTANCE_SUFFIXES:
            continue
        index.setdefault(path.stem.lower(), []).append(path)
    return {
        stem: tuple(sorted(paths, key=lambda p: (len(p.parts), str(p).lower())))
        for stem, paths in index.items()
    }


def _search_roots(plan: RunPlan) -> tuple[Path, ...]:
    family_root = BENCHMARK_ROOT / _instance_family_dir(plan.benchmark_family)
    roots: list[Path] = []
    for root in (family_root, BENCHMARK_ROOT):
        if root.exists() and root not in roots:
            roots.append(root)
    return tuple(roots)


def _candidate_instance_paths(plan: RunPlan) -> list[Path]:
    stem = plan.instance_id.lower()
    out: list[Path] = []
    seen: set[str] = set()
    for root in _search_roots(plan):
        for path in _benchmark_file_index(str(root)).get(stem, ()):
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return out


def _infer_structure_class(instance_id: str) -> str:
    up = instance_id.upper()
    if up.startswith("RC"):
        return "mixed"
    if up.startswith("R"):
        return "random"
    if up.startswith("C"):
        return "clustered"
    return "mixed"


def _parse_vehicle_capacity(lines: list[str]) -> float | None:
    for idx, line in enumerate(lines):
        norm = " ".join(line.strip().upper().split())
        if "CAPACITY" not in norm:
            continue
        for probe in lines[idx + 1 : idx + 6]:
            parts = probe.strip().split()
            if len(parts) < 2:
                continue
            try:
                return float(parts[-1])
            except ValueError:
                continue
    return None


def _parse_solomon_like(path: Path, scenario: ScenarioSpec, plan: RunPlan) -> ProblemInstance:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    data_start = None
    for idx, line in enumerate(lines):
        norm = " ".join(line.strip().upper().split())
        if "CUST NO." in norm or norm.startswith("CUSTOMER"):
            data_start = idx + 1
            break
    if data_start is None:
        raise ValueError(f"Could not find customer table header in {path}")

    rows: list[list[float]] = []
    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            vals = [float(parts[i]) for i in range(7)]
        except ValueError:
            continue
        rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric customer rows found in {path}")

    depot_row = rows[0]
    depot_xy = (float(depot_row[1]), float(depot_row[2]))
    depot_due = float(depot_row[5])
    customer_rows = rows[1:]
    if len(customer_rows) < plan.customer_count:
        raise ValueError(
            f"Instance {path.name} has only {len(customer_rows)} customers, but manifest requests {plan.customer_count}"
        )
    customer_rows = customer_rows[: plan.customer_count]

    customers: Dict[int, Customer] = {}
    for raw in customer_rows:
        cid = int(raw[0])
        x, y, demand, ready, due, service = map(float, raw[1:7])
        zone = _arc_zone(x, y, depot_xy)
        customers[cid] = Customer(
            idx=cid,
            x=x,
            y=y,
            demand=demand,
            tw_start=ready,
            tw_end=due,
            service=service,
            arc_zone=zone,
        )

    horizon = depot_due if depot_due > 0 else max(c.tw_end for c in customers.values())
    # Public Solomon/Homberger instances are single-horizon VRPTW datasets.
    # Keep all scenario shifts available, but let each span the full horizon so
    # the benchmark's native feasibility region is preserved.
    shifts = [Shift(idx=i, start=0.0, end=horizon) for i, _ in enumerate(scenario.shift_factors, start=1)]

    header_capacity = _parse_vehicle_capacity(lines)
    base_capacity = header_capacity or (max(20.0, float(depot_row[3])) if depot_row[3] > 0 else {"small": 65.0, "medium": 80.0, "large": 95.0}[plan.tier])
    vehicle_types = {
        "ICE": VehicleType(
            name="ICE",
            capacity=base_capacity,
            fixed_cost=95.0,
            energy_alpha=0.92,
            energy_beta=0.025,
            emission_factor=scenario.vehicle_factors.get("ICE", 1.0),
        ),
        "HEV": VehicleType(
            name="HEV",
            capacity=base_capacity * 0.96,
            fixed_cost=102.0,
            energy_alpha=0.80,
            energy_beta=0.021,
            emission_factor=scenario.vehicle_factors.get("HEV", 0.92),
        ),
        "EV": VehicleType(
            name="EV",
            capacity=base_capacity * 0.90,
            fixed_cost=108.0,
            energy_alpha=0.62,
            energy_beta=0.018,
            emission_factor=scenario.vehicle_factors.get("EV", 0.55),
        ),
    }
    shift_factors = {shift.idx: scenario.shift_factors[shift.idx - 1] for shift in shifts}
    try:
        source_path = str(path.relative_to(ROOT))
    except ValueError:
        source_path = str(path)
    return ProblemInstance(
        name=plan.instance_id,
        family=plan.benchmark_family,
        structure_class=plan.structure_class or _infer_structure_class(plan.instance_id),
        customer_count=len(customers),
        customers=customers,
        shifts=shifts,
        depot_xy=depot_xy,
        vehicle_types=vehicle_types,
        shift_factors=shift_factors,
        zone_factors=scenario.zone_factors,
        source_kind="real",
        source_path=source_path,
    )


def _try_load_public_instance(plan: RunPlan, scenario: ScenarioSpec) -> ProblemInstance | None:
    for path in _candidate_instance_paths(plan):
        if path.exists():
            return _parse_solomon_like(path, scenario, plan)
    return None


def _build_synthetic_problem(plan: RunPlan, scenario: ScenarioSpec) -> ProblemInstance:
    seed = _stable_seed(plan.instance_id, plan.scenario_id, plan.customer_count, plan.structure_class)
    rng = np.random.default_rng(seed)

    depot_xy = (50.0, 50.0)
    coords = _make_coords(rng, plan.customer_count, plan.structure_class)

    horizon = {"small": 960.0, "medium": 1260.0, "large": 1620.0}[plan.tier]
    shifts = []
    shift_span = horizon / len(scenario.shift_factors)
    for i, factor in enumerate(scenario.shift_factors, start=1):
        shifts.append(Shift(idx=i, start=(i - 1) * shift_span, end=i * shift_span))

    customers: Dict[int, Customer] = {}
    for i in range(1, plan.customer_count + 1):
        x, y = coords[i - 1]
        tw_start, tw_end, service = _tw_profile(rng, i, x, y, plan.tier)
        demand = float(rng.integers(2, 11 if plan.tier == "small" else 9))
        zone = _arc_zone(float(x), float(y), depot_xy)
        customers[i] = Customer(
            idx=i,
            x=float(x),
            y=float(y),
            demand=demand,
            tw_start=float(tw_start),
            tw_end=float(tw_end),
            service=float(service),
            arc_zone=zone,
        )

    base_capacity = {"small": 65.0, "medium": 80.0, "large": 95.0}[plan.tier]
    vehicle_types = {
        "ICE": VehicleType(
            name="ICE",
            capacity=base_capacity,
            fixed_cost=95.0,
            energy_alpha=0.92,
            energy_beta=0.025,
            emission_factor=scenario.vehicle_factors.get("ICE", 1.0),
        ),
        "HEV": VehicleType(
            name="HEV",
            capacity=base_capacity * 0.96,
            fixed_cost=102.0,
            energy_alpha=0.80,
            energy_beta=0.021,
            emission_factor=scenario.vehicle_factors.get("HEV", 0.92),
        ),
        "EV": VehicleType(
            name="EV",
            capacity=base_capacity * 0.90,
            fixed_cost=108.0,
            energy_alpha=0.62,
            energy_beta=0.018,
            emission_factor=scenario.vehicle_factors.get("EV", 0.55),
        ),
    }

    shift_factors = {shift.idx: scenario.shift_factors[shift.idx - 1] for shift in shifts}
    return ProblemInstance(
        name=plan.instance_id,
        family=plan.benchmark_family,
        structure_class=plan.structure_class,
        customer_count=plan.customer_count,
        customers=customers,
        shifts=shifts,
        depot_xy=depot_xy,
        vehicle_types=vehicle_types,
        shift_factors=shift_factors,
        zone_factors=scenario.zone_factors,
        source_kind="synthetic",
        source_path="",
    )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no"}


def build_problem(
    plan: RunPlan,
    scenario_map: Dict[str, ScenarioSpec] | None = None,
    require_real: bool | None = None,
) -> ProblemInstance:
    scenarios = scenario_map or load_scenarios()
    scenario = scenarios[plan.scenario_id]
    real_problem = _try_load_public_instance(plan, scenario)
    if real_problem is not None:
        return real_problem
    if require_real if require_real is not None else _env_truthy("SWEVO_REQUIRE_REAL_BENCHMARKS"):
        invalid_public_id = public_instance_id_error(plan)
        if invalid_public_id is not None:
            raise InvalidBenchmarkInstanceId(invalid_public_id)
        roots = ", ".join(str(root) for root in _search_roots(plan))
        raise FileNotFoundError(
            f"No real benchmark file found for {plan.instance_id} (family={plan.benchmark_family}) under: {roots}"
        )
    return _build_synthetic_problem(plan, scenario)
