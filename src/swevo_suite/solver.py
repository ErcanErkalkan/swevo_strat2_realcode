from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import Customer, ProblemInstance, Route, RouteMetrics, SearchStats, Solution


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def customer_pos(problem: ProblemInstance, cid: int) -> tuple[float, float]:
    c = problem.customers[cid]
    return (c.x, c.y)


def route_signature(route: Route) -> Tuple[int, str, Tuple[int, ...]]:
    return route.shift_id, route.vehicle_type, tuple(route.customers)


def evaluate_route(problem: ProblemInstance, route: Route) -> RouteMetrics:
    shift = problem.shifts[route.shift_id - 1]
    vehicle = problem.vehicle_types[route.vehicle_type]
    if not route.customers:
        return RouteMetrics()

    total_demand = sum(problem.customers[cid].demand for cid in route.customers)
    v_cap = max(0.0, total_demand - vehicle.capacity)

    metrics = RouteMetrics(cost=vehicle.fixed_cost)
    load = total_demand
    current_time = shift.start
    prev_xy = problem.depot_xy

    for cid in route.customers:
        cust = problem.customers[cid]
        dist = euclidean(prev_xy, (cust.x, cust.y))
        travel = dist
        current_time += travel
        service_start = max(current_time, cust.tw_start)
        v_tw_here = max(0.0, service_start - cust.tw_end)
        metrics.v_tw += v_tw_here
        current_time = service_start + cust.service
        zone_factor = problem.zone_factors.get(cust.arc_zone, 1.0)
        shift_factor = problem.shift_factors[route.shift_id]
        energy_arc = vehicle.energy_alpha * dist + vehicle.energy_beta * dist * load
        co2_arc = energy_arc * vehicle.emission_factor * zone_factor * shift_factor
        metrics.cost += dist
        metrics.energy += energy_arc
        metrics.co2 += co2_arc
        metrics.service_times[cid] = service_start
        load -= cust.demand
        prev_xy = (cust.x, cust.y)

    last_cid = route.customers[-1]
    last = problem.customers[last_cid]
    dist = euclidean(prev_xy, problem.depot_xy)
    current_time += dist
    zone_factor = problem.zone_factors.get(last.arc_zone, 1.0)
    shift_factor = problem.shift_factors[route.shift_id]
    energy_arc = vehicle.energy_alpha * dist + vehicle.energy_beta * dist * max(0.0, load)
    metrics.cost += dist
    metrics.energy += energy_arc
    metrics.co2 += energy_arc * vehicle.emission_factor * zone_factor * shift_factor
    overtime = max(0.0, current_time - shift.end)
    metrics.overtime = overtime
    metrics.overtime_ratio = overtime / (problem.eta * shift.length + problem.epsilon)
    metrics.v_shift = max(0.0, overtime - problem.eta * shift.length)
    metrics.v_cap = v_cap
    metrics.return_time = current_time
    return metrics


def evaluate_solution(problem: ProblemInstance, routes: Sequence[Route], references: tuple[float, float, float]) -> Solution:
    sol = Solution(routes=[Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in routes], references=references)
    total_cost = total_energy = total_co2 = 0.0
    total_ot = total_ot_ratio = 0.0
    v_cap = v_tw = v_shift = 0.0
    served = []
    for route in sol.routes:
        route.metrics = evaluate_route(problem, route)
        total_cost += route.metrics.cost
        total_energy += route.metrics.energy
        total_co2 += route.metrics.co2
        total_ot += route.metrics.overtime
        total_ot_ratio += route.metrics.overtime_ratio
        v_cap += route.metrics.v_cap
        v_tw += route.metrics.v_tw
        v_shift += route.metrics.v_shift
        served.extend(route.customers)

    counts = {cid: 0 for cid in problem.customer_ids}
    for cid in served:
        if cid in counts:
            counts[cid] += 1
    missing = sum(1 for cid, cnt in counts.items() if cnt == 0)
    dupes = sum(max(0, cnt - 1) for cnt in counts.values())
    if missing or dupes:
        v_shift += float(missing + dupes)

    c_ref, e_ref, z_ref = references
    wC, wE, wZ = problem.objective_weights
    c_t = total_cost / max(c_ref, problem.epsilon)
    e_t = total_energy / max(e_ref, problem.epsilon)
    z_t = total_co2 / max(z_ref, problem.epsilon)
    score = (wC * c_t + wE * e_t + wZ * z_t) / max(wC + wE + wZ, problem.epsilon)
    score += problem.overtime_penalty * total_ot_ratio

    sol.cost = total_cost
    sol.energy = total_energy
    sol.co2 = total_co2
    sol.overtime_sum = total_ot
    sol.overtime_ratio_sum = total_ot_ratio
    sol.v_cap = v_cap
    sol.v_tw = v_tw
    sol.v_shift = v_shift
    sol.accepted = v_cap <= 1e-9 and v_tw <= 1e-9 and v_shift <= 1e-9
    sol.strict_duty = sol.accepted and total_ot <= 1e-9
    sol.score = score if sol.accepted else score + 1000.0 + 10.0 * (v_cap + v_tw + v_shift)
    return sol


def dominance_key(sol: Solution) -> tuple[float, float, float, float]:
    return (0.0 if sol.accepted else 1.0, sol.v_cap + sol.v_tw + sol.v_shift, sol.score, sol.overtime_sum)


def _deadline_from_cap(walltime_cap_s: float | None) -> float | None:
    if walltime_cap_s is None or walltime_cap_s <= 0:
        return None
    return time.perf_counter() + float(walltime_cap_s)


def _time_exceeded(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _remaining_time(deadline: float | None) -> float:
    if deadline is None:
        return float("inf")
    return deadline - time.perf_counter()


def _phase_deadline(
    start: float,
    hard_deadline: float | None,
    walltime_cap_s: float | None,
    share: float,
) -> float | None:
    if hard_deadline is None or walltime_cap_s is None or walltime_cap_s <= 0:
        return hard_deadline
    if walltime_cap_s > 30:
        return hard_deadline
    soft_deadline = start + max(1.0, float(walltime_cap_s) * share)
    return min(hard_deadline, soft_deadline)


def _scaled_search_moves(base_moves: int, walltime_cap_s: float | None, *, minimum: int) -> int:
    if walltime_cap_s is None or walltime_cap_s <= 0:
        return base_moves
    if walltime_cap_s <= 15:
        return max(minimum, base_moves // 4)
    if walltime_cap_s <= 30:
        return max(minimum, base_moves // 3)
    if walltime_cap_s <= 60:
        return max(minimum, (base_moves * 3) // 4)
    return base_moves


def non_dominated_insert(archive: List[Solution], cand: Solution, max_size: int = 128) -> List[Solution]:
    if not cand.accepted:
        return archive
    keep: List[Solution] = []
    dominated = False
    for cur in archive:
        if (
            cur.cost <= cand.cost and cur.energy <= cand.energy and cur.co2 <= cand.co2
            and (cur.cost < cand.cost or cur.energy < cand.energy or cur.co2 < cand.co2)
        ):
            dominated = True
            keep.append(cur)
        elif (
            cand.cost <= cur.cost and cand.energy <= cur.energy and cand.co2 <= cur.co2
            and (cand.cost < cur.cost or cand.energy < cur.energy or cand.co2 < cur.co2)
        ):
            continue
        else:
            keep.append(cur)
    if not dominated:
        keep.append(cand.copy())
    keep.sort(key=lambda s: (s.cost, s.energy, s.co2))
    return keep[:max_size]


def _temporal_clusters(problem: ProblemInstance, k: int) -> List[List[int]]:
    items = []
    for cid, cust in problem.customers.items():
        mid = 0.5 * (cust.tw_start + cust.tw_end)
        items.append((cid, cust.x, cust.y, mid))
    arr = np.array([[x, y, mid / max(problem.shifts[-1].end, 1.0)] for _, x, y, mid in items], dtype=float)
    arr[:, :2] /= 100.0
    k = max(2, min(k, len(items)))
    seed_idx = np.linspace(0, len(items) - 1, k, dtype=int)
    centers = arr[seed_idx].copy()
    for _ in range(10):
        d2 = ((arr[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = arr[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    groups = [[] for _ in range(k)]
    for (cid, *_), lbl in zip(items, labels):
        groups[int(lbl)].append(cid)
    for group in groups:
        group.sort(key=lambda cid: (0.5 * (problem.customers[cid].tw_start + problem.customers[cid].tw_end), cid))
    groups = [g for g in groups if g]
    groups.sort(key=lambda g: 0.5 * (problem.customers[g[0]].tw_start + problem.customers[g[0]].tw_end))
    return groups


def seed_permutation(problem: ProblemInstance, rng: random.Random) -> List[int]:
    k = 4 if problem.customer_count <= 120 else 6 if problem.customer_count <= 240 else 8
    groups = _temporal_clusters(problem, k)
    ordered: List[int] = []
    for group in groups:
        group.sort(key=lambda cid: (
            next(abs(problem.customers[cid].x - problem.depot_xy[0]) + abs(problem.customers[cid].y - problem.depot_xy[1]) for _ in [0]),
            problem.customers[cid].tw_start,
            cid,
        ))
        ordered.extend(group)
    return ordered


def build_seed_references(problem: ProblemInstance) -> tuple[float, float, float]:
    rng = random.Random(12345 + problem.customer_count)
    perm = seed_permutation(problem, rng)
    routes = decode_permutation(problem, perm, (1.0, 1.0, 1.0), allow_infeasible_fallback=True)
    sol = evaluate_solution(problem, routes, (1.0, 1.0, 1.0))
    return (max(sol.cost, 1.0), max(sol.energy, 1.0), max(sol.co2, 1.0))


def _route_totals(routes: Sequence[Route]) -> tuple[float, float, float, float, float, float, float, float]:
    return (
        sum(route.metrics.cost for route in routes),
        sum(route.metrics.energy for route in routes),
        sum(route.metrics.co2 for route in routes),
        sum(route.metrics.overtime for route in routes),
        sum(route.metrics.overtime_ratio for route in routes),
        sum(route.metrics.v_cap for route in routes),
        sum(route.metrics.v_tw for route in routes),
        sum(route.metrics.v_shift for route in routes),
    )


def _coverage_penalty_after_insertion(problem: ProblemInstance, routes: Sequence[Route], cid: int) -> float:
    seen: set[int] = set()
    dupes = 0
    cid_present = False
    for route in routes:
        for existing in route.customers:
            if existing == cid:
                cid_present = True
            if existing in seen:
                dupes += 1
            else:
                seen.add(existing)
    adds_unique = 0 if cid_present else 1
    missing = max(0, problem.customer_count - len(seen) - adds_unique)
    return float(dupes + missing)


def _assemble_solution_from_route_update(
    problem: ProblemInstance,
    routes: Sequence[Route],
    references: tuple[float, float, float],
    base_totals: tuple[float, float, float, float, float, float, float, float],
    coverage_penalty: float,
    replace_idx: int | None,
    new_route: Route,
) -> Solution:
    base_cost, base_energy, base_co2, base_ot, base_ot_ratio, base_v_cap, base_v_tw, base_v_shift = base_totals
    old_metrics = routes[replace_idx].metrics if replace_idx is not None else None
    if old_metrics is not None:
        total_cost = base_cost - old_metrics.cost + new_route.metrics.cost
        total_energy = base_energy - old_metrics.energy + new_route.metrics.energy
        total_co2 = base_co2 - old_metrics.co2 + new_route.metrics.co2
        total_ot = base_ot - old_metrics.overtime + new_route.metrics.overtime
        total_ot_ratio = base_ot_ratio - old_metrics.overtime_ratio + new_route.metrics.overtime_ratio
        v_cap = base_v_cap - old_metrics.v_cap + new_route.metrics.v_cap
        v_tw = base_v_tw - old_metrics.v_tw + new_route.metrics.v_tw
        route_v_shift = base_v_shift - old_metrics.v_shift + new_route.metrics.v_shift
        cand_routes = list(routes)
        cand_routes[replace_idx] = new_route
    else:
        total_cost = base_cost + new_route.metrics.cost
        total_energy = base_energy + new_route.metrics.energy
        total_co2 = base_co2 + new_route.metrics.co2
        total_ot = base_ot + new_route.metrics.overtime
        total_ot_ratio = base_ot_ratio + new_route.metrics.overtime_ratio
        v_cap = base_v_cap + new_route.metrics.v_cap
        v_tw = base_v_tw + new_route.metrics.v_tw
        route_v_shift = base_v_shift + new_route.metrics.v_shift
        cand_routes = list(routes) + [new_route]

    v_shift = route_v_shift + coverage_penalty
    c_ref, e_ref, z_ref = references
    wC, wE, wZ = problem.objective_weights
    c_t = total_cost / max(c_ref, problem.epsilon)
    e_t = total_energy / max(e_ref, problem.epsilon)
    z_t = total_co2 / max(z_ref, problem.epsilon)
    score = (wC * c_t + wE * e_t + wZ * z_t) / max(wC + wE + wZ, problem.epsilon)
    score += problem.overtime_penalty * total_ot_ratio
    accepted = v_cap <= 1e-9 and v_tw <= 1e-9 and v_shift <= 1e-9
    if not accepted:
        score += 1000.0 + 10.0 * (v_cap + v_tw + v_shift)
    return Solution(
        routes=cand_routes,
        references=references,
        score=score,
        cost=total_cost,
        energy=total_energy,
        co2=total_co2,
        overtime_sum=total_ot,
        overtime_ratio_sum=total_ot_ratio,
        v_cap=v_cap,
        v_tw=v_tw,
        v_shift=v_shift,
        accepted=accepted,
        strict_duty=accepted and total_ot <= 1e-9,
    )


def _candidate_insertions(problem: ProblemInstance, routes: Sequence[Route], cid: int, references: tuple[float, float, float]) -> List[Solution]:
    proposals: List[Solution] = []
    vehicle_names = ["HEV", "EV", "ICE"]
    cust = problem.customers[cid]
    cust_mid = 0.5 * (cust.tw_start + cust.tw_end)
    base_totals = _route_totals(routes)
    coverage_penalty = _coverage_penalty_after_insertion(problem, routes, cid)

    scored_routes = []
    for ridx, route in enumerate(routes):
        if route.customers:
            mids = [0.5 * (problem.customers[x].tw_start + problem.customers[x].tw_end) for x in route.customers]
            mean_mid = sum(mids) / len(mids)
            coords = np.array([[problem.customers[x].x, problem.customers[x].y] for x in route.customers], dtype=float)
            cx, cy = coords.mean(axis=0)
            geo = euclidean((cust.x, cust.y), (float(cx), float(cy)))
            score = 0.65 * geo + 0.35 * abs(cust_mid - mean_mid) / max(problem.shifts[-1].end, 1.0)
        else:
            score = 1e9
        scored_routes.append((score, ridx, route))
    scored_routes.sort(key=lambda t: t[0])
    route_limit = 6 if problem.customer_count <= 120 else 8 if problem.customer_count <= 240 else 10
    for _, ridx, route in scored_routes[:route_limit]:
        pos_candidates = {0, len(route.customers)}
        if route.customers:
            nearest_idx = min(range(len(route.customers)), key=lambda i: euclidean((cust.x, cust.y), customer_pos(problem, route.customers[i])))
            pos_candidates.update({nearest_idx, nearest_idx + 1})
        for pos in sorted(p for p in pos_candidates if 0 <= p <= len(route.customers)):
            new_customers = list(route.customers)
            new_customers.insert(pos, cid)
            new_route = Route(route.shift_id, route.vehicle_type, new_customers)
            new_route.metrics = evaluate_route(problem, new_route)
            proposals.append(
                _assemble_solution_from_route_update(
                    problem,
                    routes,
                    references,
                    base_totals,
                    coverage_penalty,
                    ridx,
                    new_route,
                )
            )
    for shift in problem.shifts:
        for vehicle_name in vehicle_names[:2]:
            new_route = Route(shift.idx, vehicle_name, [cid])
            new_route.metrics = evaluate_route(problem, new_route)
            proposals.append(
                _assemble_solution_from_route_update(
                    problem,
                    routes,
                    references,
                    base_totals,
                    coverage_penalty,
                    None,
                    new_route,
                )
            )
    return proposals


def decode_permutation(
    problem: ProblemInstance,
    permutation: Sequence[int],
    references: tuple[float, float, float],
    allow_infeasible_fallback: bool = True,
) -> List[Route]:
    routes: List[Route] = []
    for cid in permutation:
        proposals = _candidate_insertions(problem, routes, cid, references) if routes else []
        if not proposals:
            best = None
            for shift in problem.shifts:
                for vehicle_name in problem.vehicle_types:
                    cand_routes = [Route(shift.idx, vehicle_name, [cid])]
                    sol = evaluate_solution(problem, cand_routes, references)
                    if best is None or dominance_key(sol) < dominance_key(best):
                        best = sol
            assert best is not None
            routes = best.routes
            continue
        feasible = [p for p in proposals if p.accepted]
        if feasible:
            feasible.sort(key=lambda s: (s.score, s.overtime_sum, len(s.routes)))
            routes = feasible[0].routes
        else:
            proposals.sort(key=dominance_key)
            if allow_infeasible_fallback:
                routes = proposals[0].routes
            else:
                # open a fresh route in the shift nearest to customer midpoint
                cust = problem.customers[cid]
                shift = min(problem.shifts, key=lambda sh: abs((cust.tw_start + cust.tw_end) * 0.5 - (sh.start + sh.end) * 0.5))
                routes.append(Route(shift.idx, "HEV", [cid]))
    return routes


def permutation_from_solution(sol: Solution) -> List[int]:
    out: List[int] = []
    for route in sorted(sol.routes, key=lambda r: (r.shift_id, r.metrics.return_time, len(r.customers))):
        out.extend(route.customers)
    return out


def permutation_from_keys(keys: np.ndarray) -> List[int]:
    idx = np.arange(1, keys.shape[0] + 1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    return idx[order].tolist()


def keys_from_permutation(perm: Sequence[int], rng: random.Random) -> np.ndarray:
    n = len(perm)
    keys = np.zeros((n, 2), dtype=float)
    for rank, cid in enumerate(perm):
        keys[cid - 1, 0] = (rank + 1) / (n + 1) + rng.uniform(-1e-4, 1e-4)
        keys[cid - 1, 1] = rng.random()
    return np.clip(keys, 0.0, 1.0 - 1e-9)


def route_boundary_customers(problem: ProblemInstance, sol: Solution) -> List[int]:
    out: List[int] = []
    for route in sol.routes:
        shift = next(s for s in problem.shifts if s.idx == route.shift_id)
        for cid, start in route.metrics.service_times.items():
            if abs(start - shift.start) < 0.12 * shift.length or abs(shift.end - start) < 0.18 * shift.length:
                out.append(cid)
    return sorted(set(out))


def try_improve_with_local_search(
    problem: ProblemInstance,
    sol: Solution,
    references: tuple[float, float, float],
    max_moves: int = 32,
    deadline: float | None = None,
) -> Solution:
    best = sol.copy()
    moves = 0
    improved = True
    while improved and moves < max_moves:
        if _time_exceeded(deadline):
            break
        improved = False
        # relocate
        for ridx, route in enumerate(best.routes):
            if moves >= max_moves:
                break
            for pos, cid in enumerate(list(route.customers)):
                for tridx, troute in enumerate(best.routes):
                    for tpos in range(len(troute.customers) + 1):
                        if _time_exceeded(deadline):
                            return best
                        if ridx == tridx and (tpos == pos or tpos == pos + 1):
                            continue
                        cand_routes = [Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in best.routes]
                        cand_routes[ridx].customers.pop(pos)
                        if not cand_routes[ridx].customers:
                            cand_routes.pop(ridx)
                            if tridx > ridx:
                                tridx2 = tridx - 1
                            else:
                                tridx2 = tridx
                        else:
                            tridx2 = tridx
                        cand_routes[tridx2].customers.insert(min(tpos, len(cand_routes[tridx2].customers)), cid)
                        cand = evaluate_solution(problem, cand_routes, references)
                        moves += 1
                        if dominance_key(cand) < dominance_key(best):
                            best = cand
                            improved = True
                            break
                    if improved or moves >= max_moves:
                        break
                if improved or moves >= max_moves:
                    break
            if improved or moves >= max_moves:
                break
        if improved:
            continue
        # swap
        for ridx in range(len(best.routes)):
            for sidx in range(ridx, len(best.routes)):
                ra = best.routes[ridx]
                rb = best.routes[sidx]
                for ia, ca in enumerate(ra.customers):
                    for ib, cb in enumerate(rb.customers):
                        if _time_exceeded(deadline):
                            return best
                        if ridx == sidx and ia == ib:
                            continue
                        cand_routes = [Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in best.routes]
                        cand_routes[ridx].customers[ia], cand_routes[sidx].customers[ib] = cb, ca
                        cand = evaluate_solution(problem, cand_routes, references)
                        moves += 1
                        if dominance_key(cand) < dominance_key(best):
                            best = cand
                            improved = True
                            break
                    if improved or moves >= max_moves:
                        break
                if improved or moves >= max_moves:
                    break
            if improved or moves >= max_moves:
                break
    return best


def bounded_repair(
    problem: ProblemInstance,
    sol: Solution,
    references: tuple[float, float, float],
    stats: SearchStats,
    max_attempts: int = 16,
    deadline: float | None = None,
) -> Solution:
    current = sol.copy()
    for _ in range(max_attempts):
        if _time_exceeded(deadline):
            break
        if current.accepted:
            break
        stats.n_repair_attempts += 1
        bad_routes = sorted(current.routes, key=lambda r: (r.metrics.v_tw + r.metrics.v_shift + r.metrics.v_cap, len(r.customers)), reverse=True)
        if not bad_routes:
            break
        chosen = bad_routes[0]
        if not chosen.customers:
            break
        removable = sorted(chosen.customers, key=lambda cid: current.routes[current.routes.index(chosen)].metrics.service_times.get(cid, 0.0), reverse=True)
        cid = removable[0]
        cand_routes = [Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in current.routes]
        for route in cand_routes:
            if route_signature(route) == route_signature(chosen):
                route.customers.remove(cid)
                break
        cand_routes = [r for r in cand_routes if r.customers]
        partial = evaluate_solution(problem, cand_routes, references)
        reinserts = _candidate_insertions(problem, partial.routes, cid, references)
        reinserts.sort(key=dominance_key)
        new_sol = reinserts[0] if reinserts else partial
        if dominance_key(new_sol) <= dominance_key(current):
            current = new_sol
            stats.n_repair_success += 1
        else:
            break
    return current


def boundary_lns(
    problem: ProblemInstance,
    sol: Solution,
    references: tuple[float, float, float],
    rng: random.Random,
    destroy_frac: float = 0.18,
    ls_moves: int = 24,
    deadline: float | None = None,
) -> Solution:
    boundary = route_boundary_customers(problem, sol)
    if not boundary:
        boundary = permutation_from_solution(sol)
    k = max(1, int(len(boundary) * destroy_frac))
    removed = set(rng.sample(boundary, min(k, len(boundary))))
    base_routes = []
    removed_list = []
    for route in sol.routes:
        kept = [cid for cid in route.customers if cid not in removed]
        if kept:
            base_routes.append(Route(route.shift_id, route.vehicle_type, kept))
        removed_list.extend([cid for cid in route.customers if cid in removed])
    current = evaluate_solution(problem, base_routes, references)
    for cid in sorted(removed_list, key=lambda c: problem.customers[c].tw_start):
        if _time_exceeded(deadline):
            return current
        proposals = _candidate_insertions(problem, current.routes, cid, references)
        proposals.sort(key=dominance_key)
        current = proposals[0]
    current = try_improve_with_local_search(problem, current, references, max_moves=ls_moves, deadline=deadline)
    return current


def _solution_without_customers(
    problem: ProblemInstance,
    sol: Solution,
    removed: set[int],
    references: tuple[float, float, float],
) -> Solution:
    base_routes = []
    for route in sol.routes:
        kept = [cid for cid in route.customers if cid not in removed]
        if kept:
            base_routes.append(Route(route.shift_id, route.vehicle_type, kept))
    return evaluate_solution(problem, base_routes, references)


def _repair_value(sol: Solution) -> float:
    return sol.score + 1e-4 * sol.overtime_sum + 1e-5 * len(sol.routes)


def _regret_repair(
    problem: ProblemInstance,
    partial: Solution,
    removed: Sequence[int],
    references: tuple[float, float, float],
    deadline: float | None = None,
) -> Solution:
    current = partial
    remaining = list(dict.fromkeys(removed))
    while remaining:
        if _time_exceeded(deadline):
            break
        choice_idx = 0
        choice_sol: Solution | None = None
        choice_best = float("inf")
        choice_regret = -float("inf")
        candidate_ids = remaining if len(remaining) <= 10 else remaining[:6] + remaining[-4:]
        for cid in candidate_ids:
            proposals = _candidate_insertions(problem, current.routes, cid, references)
            if not proposals:
                continue
            proposals.sort(key=dominance_key)
            best_sol = proposals[0]
            best_val = _repair_value(best_sol)
            second_val = _repair_value(proposals[1]) if len(proposals) > 1 else best_val + 1.0
            regret = second_val - best_val
            if regret > choice_regret + 1e-9 or (abs(regret - choice_regret) <= 1e-9 and best_val < choice_best):
                choice_regret = regret
                choice_best = best_val
                choice_sol = best_sol
                choice_idx = remaining.index(cid)
        if choice_sol is None:
            break
        current = choice_sol
        remaining.pop(choice_idx)
    return current


def _route_segment_destroy(problem: ProblemInstance, sol: Solution, rng: random.Random, count: int) -> set[int]:
    if not sol.routes:
        return set()
    horizon = max(problem.shifts[-1].end, 1.0)
    scored_routes: list[tuple[float, Route]] = []
    for route in sol.routes:
        if not route.customers:
            continue
        critical_mass = 0.0
        for cid in route.customers:
            start = route.metrics.service_times.get(cid)
            if start is None:
                continue
            slack = max(0.0, problem.customers[cid].tw_end - start)
            critical_mass += 1.0 / max(slack / horizon, 1e-3)
        scored_routes.append((critical_mass + 0.1 * len(route.customers), route))
    if not scored_routes:
        return set()
    scored_routes.sort(key=lambda item: item[0], reverse=True)
    route = scored_routes[0][1]
    if len(route.customers) <= count:
        return set(route.customers)
    critical = critical_route_customers(problem, sol)
    if critical:
        focus = next((cid for cid in route.customers if cid in critical), route.customers[len(route.customers) // 2])
        center = route.customers.index(focus)
    else:
        center = rng.randrange(len(route.customers))
    span = max(1, min(count, len(route.customers)))
    start = max(0, min(len(route.customers) - span, center - span // 2))
    return set(route.customers[start : start + span])


def _select_destroy_customers(problem: ProblemInstance, sol: Solution, rng: random.Random, count: int) -> tuple[str, list[int]]:
    perm = permutation_from_solution(sol)
    if not perm:
        return "random", []
    count = max(1, min(count, len(perm)))
    op = rng.choices(
        ["critical", "related", "segment", "random"],
        weights=[0.35, 0.25, 0.25, 0.15],
        k=1,
    )[0]
    if op == "critical":
        critical = critical_route_customers(problem, sol)
        if critical:
            ordered = sorted(
                critical,
                key=lambda cid: problem.customers[cid].tw_end - next(
                    (route.metrics.service_times.get(cid, problem.customers[cid].tw_end) for route in sol.routes if cid in route.metrics.service_times),
                    problem.customers[cid].tw_end,
                ),
            )
            picks = ordered[:count]
            if len(picks) < count:
                remainder = [cid for cid in perm if cid not in picks]
                picks.extend(remainder[: count - len(picks)])
            return op, picks
    if op == "related":
        anchor = rng.choice(critical_route_customers(problem, sol) or perm)
        ax, ay = customer_pos(problem, anchor)
        amid = 0.5 * (problem.customers[anchor].tw_start + problem.customers[anchor].tw_end)
        related = sorted(
            perm,
            key=lambda cid: (
                0.7 * euclidean((ax, ay), customer_pos(problem, cid))
                + 0.3 * abs((0.5 * (problem.customers[cid].tw_start + problem.customers[cid].tw_end)) - amid),
                cid,
            ),
        )
        return op, related[:count]
    if op == "segment":
        seg = list(_route_segment_destroy(problem, sol, rng, count))
        if seg:
            return op, seg[:count]
    return "random", rng.sample(perm, count)


def route_alns_endgame(
    problem: ProblemInstance,
    seed_sol: Solution,
    references: tuple[float, float, float],
    rng: random.Random,
    stats: SearchStats,
    iterations: int,
    repair_budget: int,
    ls_moves: int,
    polish_moves: int = 0,
    deadline: float | None = None,
) -> tuple[Solution, Solution, list[Solution]]:
    current = seed_sol.copy()
    best = seed_sol.copy()
    generated: list[Solution] = []
    temperature = 0.028
    destroy_count = max(3, int(problem.customer_count * (0.08 if problem.customer_count <= 120 else 0.06)))
    for _ in range(iterations):
        if _time_exceeded(deadline):
            break
        op, removed = _select_destroy_customers(problem, current, rng, destroy_count)
        if not removed:
            break
        partial = _solution_without_customers(problem, current, set(removed), references)
        cand = _regret_repair(problem, partial, sorted(removed, key=lambda cid: problem.customers[cid].tw_start), references, deadline=deadline)
        cand = bounded_repair(problem, cand, references, stats, max_attempts=max(4, repair_budget), deadline=deadline)
        cand = try_improve_with_local_search(problem, cand, references, max_moves=ls_moves, deadline=deadline)
        if polish_moves > 0:
            cand = deep_route_polish(problem, cand, references, max_moves=polish_moves, deadline=deadline)
        generated.append(cand)
        delta = cand.score - current.score
        if dominance_key(cand) < dominance_key(current) or rng.random() < math.exp(-max(0.0, delta) / max(temperature, 1e-6)):
            current = cand
        if dominance_key(cand) < dominance_key(best):
            best = cand
        temperature *= 0.93
        if best.accepted and current.accepted and op == "critical":
            destroy_count = max(2, destroy_count - 1)
    return best, current, generated


def critical_route_customers(problem: ProblemInstance, sol: Solution) -> List[int]:
    critical = route_boundary_customers(problem, sol)
    horizon = max(problem.shifts[-1].end, 1.0)
    for route in sol.routes:
        if not route.customers:
            continue
        critical.extend(route.customers[:2])
        critical.extend(route.customers[-2:])
        for cid in route.customers:
            start = route.metrics.service_times.get(cid)
            if start is None:
                continue
            slack = problem.customers[cid].tw_end - start
            if slack < 0.08 * horizon:
                critical.append(cid)
    return sorted(set(critical))


def _swap_customers_in_perm(perm: list[int], a: int, b: int) -> None:
    ia, ib = perm.index(a), perm.index(b)
    perm[ia], perm[ib] = perm[ib], perm[ia]


def ils_guided_permutation_from_solution(problem: ProblemInstance, sol: Solution, rng: random.Random) -> List[int]:
    perm = permutation_from_solution(sol)
    n = len(perm)
    if n < 2:
        return perm

    border = route_boundary_customers(problem, sol)
    critical = critical_route_customers(problem, sol)
    if len(border) >= 2:
        picks = rng.sample(border, min(len(border), 4 if n >= 40 else 2))
        while len(picks) >= 2:
            a = picks.pop()
            b = picks.pop()
            _swap_customers_in_perm(perm, a, b)
    else:
        swaps = 2 if n >= 80 else 1
        for _ in range(swaps):
            i, j = rng.sample(range(n), 2)
            perm[i], perm[j] = perm[j], perm[i]

    if critical:
        mover = rng.choice(critical)
        src = perm.index(mover)
        radius = max(3, min(10, n // 12))
        dst = min(n - 1, max(0, src + rng.randint(-radius, radius)))
        customer = perm.pop(src)
        perm.insert(dst, customer)

    if n >= 6 and rng.random() < 0.45:
        focus = critical if len(critical) >= 2 else perm
        a, b = sorted(perm.index(cid) for cid in rng.sample(focus, 2))
        if b - a >= 2:
            span_limit = max(3, min(8, n // 10))
            if b - a > span_limit:
                b = a + span_limit
            perm[a : b + 1] = reversed(perm[a : b + 1])
    return perm


def perturbed_permutation_from_solution(problem: ProblemInstance, sol: Solution, rng: random.Random) -> List[int]:
    base_perm = permutation_from_solution(sol)
    if len(base_perm) < 2:
        return base_perm
    focus = critical_route_customers(problem, sol)
    if len(focus) >= 2:
        a, b = rng.sample(focus, 2)
        ia, ib = base_perm.index(a), base_perm.index(b)
        base_perm[ia], base_perm[ib] = base_perm[ib], base_perm[ia]
    else:
        swaps = max(2, problem.customer_count // 35)
        for _ in range(swaps):
            i, j = rng.sample(range(problem.customer_count), 2)
            base_perm[i], base_perm[j] = base_perm[j], base_perm[i]
    return base_perm


def deep_route_polish(
    problem: ProblemInstance,
    sol: Solution,
    references: tuple[float, float, float],
    max_moves: int = 32,
    deadline: float | None = None,
) -> Solution:
    def route_center(route: Route) -> tuple[float, float]:
        if not route.customers:
            return problem.depot_xy
        xs = [problem.customers[cid].x for cid in route.customers]
        ys = [problem.customers[cid].y for cid in route.customers]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def route_pairs(candidate: Solution, limit: int = 10) -> list[tuple[int, int]]:
        horizon = max(problem.shifts[-1].end, 1.0)
        centers = [route_center(route) for route in candidate.routes]
        mids = []
        for route in candidate.routes:
            if route.customers:
                vals = [0.5 * (problem.customers[cid].tw_start + problem.customers[cid].tw_end) for cid in route.customers]
                mids.append(sum(vals) / len(vals))
            else:
                mids.append(0.5 * horizon)
        scored = []
        for ridx in range(len(candidate.routes)):
            for jdx in range(ridx + 1, len(candidate.routes)):
                spatial = euclidean(centers[ridx], centers[jdx])
                temporal = abs(mids[ridx] - mids[jdx]) / horizon
                scored.append((spatial + 12.0 * temporal, ridx, jdx))
        scored.sort(key=lambda item: item[0])
        return [(ridx, jdx) for _, ridx, jdx in scored[:limit]]

    best = sol.copy()
    moves = 0
    improved = True
    while improved and moves < max_moves:
        if _time_exceeded(deadline):
            break
        improved = False
        # Intra-route 2-opt
        for ridx, route in enumerate(best.routes):
            n = len(route.customers)
            if n < 4:
                continue
            for i in range(n - 2):
                for j in range(i + 2, n + 1):
                    if _time_exceeded(deadline):
                        return best
                    cand_routes = [Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in best.routes]
                    cand_routes[ridx].customers = (
                        cand_routes[ridx].customers[:i]
                        + list(reversed(cand_routes[ridx].customers[i:j]))
                        + cand_routes[ridx].customers[j:]
                    )
                    cand = evaluate_solution(problem, cand_routes, references)
                    moves += 1
                    if dominance_key(cand) < dominance_key(best):
                        best = cand
                        improved = True
                        break
                if improved or moves >= max_moves:
                    break
            if improved or moves >= max_moves:
                break
        if improved:
            continue
        # Same-route segment reinsertion
        for ridx, route in enumerate(best.routes):
            n = len(route.customers)
            if n < 4:
                continue
            for seg_len in (2, 3):
                if n <= seg_len:
                    continue
                for start in range(n - seg_len + 1):
                    segment = route.customers[start : start + seg_len]
                    remaining = route.customers[:start] + route.customers[start + seg_len :]
                    for tpos in range(len(remaining) + 1):
                        if _time_exceeded(deadline):
                            return best
                        if tpos == start:
                            continue
                        reordered = remaining[:tpos] + segment + remaining[tpos:]
                        cand_routes = [Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in best.routes]
                        cand_routes[ridx].customers = reordered
                        cand = evaluate_solution(problem, cand_routes, references)
                        moves += 1
                        if dominance_key(cand) < dominance_key(best):
                            best = cand
                            improved = True
                            break
                    if improved or moves >= max_moves:
                        break
                if improved or moves >= max_moves:
                    break
            if improved or moves >= max_moves:
                break
        if improved:
            continue
        # Inter-route segment relocate
        for ridx, jdx in route_pairs(best):
            directions = [(ridx, jdx), (jdx, ridx)]
            for src_idx, dst_idx in directions:
                src = best.routes[src_idx]
                dst = best.routes[dst_idx]
                for seg_len in (1, 2):
                    if len(src.customers) <= seg_len:
                        continue
                    dst_positions = {0, len(dst.customers)}
                    if dst.customers:
                        anchor = src.customers[0]
                        nearest = min(
                            range(len(dst.customers)),
                            key=lambda pos: euclidean(customer_pos(problem, anchor), customer_pos(problem, dst.customers[pos])),
                        )
                        dst_positions.update({nearest, nearest + 1})
                    for start in range(len(src.customers) - seg_len + 1):
                        segment = src.customers[start : start + seg_len]
                        base_src = src.customers[:start] + src.customers[start + seg_len :]
                        for pos in sorted(p for p in dst_positions if 0 <= p <= len(dst.customers)):
                            if _time_exceeded(deadline):
                                return best
                            cand_routes = [Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in best.routes]
                            cand_routes[src_idx].customers = list(base_src)
                            cand_routes[dst_idx].customers = cand_routes[dst_idx].customers[:pos] + list(segment) + cand_routes[dst_idx].customers[pos:]
                            cand_routes = [route for route in cand_routes if route.customers]
                            cand = evaluate_solution(problem, cand_routes, references)
                            moves += 1
                            if dominance_key(cand) < dominance_key(best):
                                best = cand
                                improved = True
                                break
                        if improved or moves >= max_moves:
                            break
                    if improved or moves >= max_moves:
                        break
                if improved or moves >= max_moves:
                    break
            if improved or moves >= max_moves:
                break
        if improved:
            continue
        # Small cross-exchange between nearby routes
        for ridx, jdx in route_pairs(best):
            ra = best.routes[ridx]
            rb = best.routes[jdx]
            for la, lb in ((1, 1), (2, 1), (1, 2)):
                if len(ra.customers) < la or len(rb.customers) < lb:
                    continue
                for ia in range(len(ra.customers) - la + 1):
                    seg_a = ra.customers[ia : ia + la]
                    rest_a = ra.customers[:ia] + ra.customers[ia + la :]
                    for ib in range(len(rb.customers) - lb + 1):
                        if _time_exceeded(deadline):
                            return best
                        seg_b = rb.customers[ib : ib + lb]
                        rest_b = rb.customers[:ib] + rb.customers[ib + lb :]
                        cand_routes = [Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in best.routes]
                        cand_routes[ridx].customers = rest_a[:ia] + list(seg_b) + rest_a[ia:]
                        cand_routes[jdx].customers = rest_b[:ib] + list(seg_a) + rest_b[ib:]
                        cand = evaluate_solution(problem, cand_routes, references)
                        moves += 1
                        if dominance_key(cand) < dominance_key(best):
                            best = cand
                            improved = True
                            break
                    if improved or moves >= max_moves:
                        break
                if improved or moves >= max_moves:
                    break
            if improved or moves >= max_moves:
                break
    return best


def elite_route_perturbation(
    problem: ProblemInstance,
    sol: Solution,
    references: tuple[float, float, float],
    rng: random.Random,
    stats: SearchStats,
    repair_budget: int,
    ls_moves: int,
    polish_moves: int = 0,
    deadline: float | None = None,
) -> Solution:
    current = sol.copy()
    best = sol.copy()
    temperature = 0.03
    for _ in range(4):
        if _time_exceeded(deadline):
            break
        cand_perm = perturbed_permutation_from_solution(problem, current, rng)
        cand = evaluate_solution(problem, decode_permutation(problem, cand_perm, references), references)
        cand = bounded_repair(problem, cand, references, stats, max_attempts=max(4, repair_budget // 2), deadline=deadline)
        cand = try_improve_with_local_search(problem, cand, references, max_moves=ls_moves, deadline=deadline)
        if polish_moves > 0:
            cand = deep_route_polish(problem, cand, references, max_moves=polish_moves, deadline=deadline)
        delta = cand.score - current.score
        if dominance_key(cand) < dominance_key(current) or rng.random() < math.exp(-max(0.0, delta) / max(temperature, 1e-6)):
            current = cand
        if dominance_key(cand) < dominance_key(best):
            best = cand
        temperature *= 0.85
    return best


def trajectory_intensification(
    problem: ProblemInstance,
    seed_sol: Solution,
    references: tuple[float, float, float],
    rng: random.Random,
    stats: SearchStats,
    iterations: int,
    repair_budget: int,
    ls_moves: int,
    polish_moves: int = 0,
    deadline: float | None = None,
) -> tuple[Solution, Solution, list[Solution]]:
    current = seed_sol.copy()
    best = seed_sol.copy()
    generated: list[Solution] = []
    temperature = 0.035
    for step in range(iterations):
        if _time_exceeded(deadline):
            break
        if step % 4 == 0:
            cand = boundary_lns(
                problem,
                current,
                references,
                rng,
                destroy_frac=0.14 if problem.customer_count <= 120 else 0.10,
                ls_moves=max(18, ls_moves // 2),
                deadline=deadline,
            )
        elif step % 4 == 2 and current.accepted:
            t_best, t_current, generated = route_alns_endgame(
                problem,
                current,
                references,
                rng,
                stats,
                iterations=1,
                repair_budget=max(4, repair_budget // 2),
                ls_moves=max(14, ls_moves),
                polish_moves=max(0, polish_moves // 2),
                deadline=deadline,
            )
            cand = t_best if generated else t_current
        else:
            perm = perturbed_permutation_from_solution(problem, current, rng)
            cand = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
            if not cand.accepted:
                cand = bounded_repair(problem, cand, references, stats, max_attempts=max(4, repair_budget // 2), deadline=deadline)
            cand = try_improve_with_local_search(problem, cand, references, max_moves=ls_moves, deadline=deadline)
        if polish_moves > 0:
            cand = deep_route_polish(problem, cand, references, max_moves=polish_moves, deadline=deadline)
        generated.append(cand)
        delta = cand.score - current.score
        if dominance_key(cand) < dominance_key(current) or rng.random() < math.exp(-max(0.0, delta) / max(temperature, 1e-6)):
            current = cand
        if dominance_key(cand) < dominance_key(best):
            best = cand
        temperature *= 0.90
    return best, current, generated


def incumbent_ils_burst(
    problem: ProblemInstance,
    seed_sol: Solution,
    references: tuple[float, float, float],
    rng: random.Random,
    stats: SearchStats,
    iterations: int,
    repair_budget: int,
    ls_moves: int,
    polish_moves: int = 0,
    deadline: float | None = None,
) -> tuple[Solution, Solution, list[Solution]]:
    current = seed_sol.copy()
    best = seed_sol.copy()
    generated: list[Solution] = []
    temperature = 0.05
    for _ in range(iterations):
        if _time_exceeded(deadline):
            break
        perm = permutation_from_solution(current)
        border = route_boundary_customers(problem, current)
        if len(border) >= 2:
            a, b = rng.sample(border, 2)
            _swap_customers_in_perm(perm, a, b)
        else:
            for _ in range(max(2, problem.customer_count // 35)):
                i, j = rng.sample(range(problem.customer_count), 2)
                perm[i], perm[j] = perm[j], perm[i]
        cand = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
        if not cand.accepted:
            cand = bounded_repair(problem, cand, references, stats, max_attempts=max(4, repair_budget // 2), deadline=deadline)
        cand = try_improve_with_local_search(problem, cand, references, max_moves=ls_moves, deadline=deadline)
        if polish_moves > 0:
            cand = deep_route_polish(problem, cand, references, max_moves=polish_moves, deadline=deadline)
        generated.append(cand)
        delta = cand.score - current.score
        if dominance_key(cand) < dominance_key(current) or rng.random() < math.exp(-max(0.0, delta) / max(temperature, 1e-6)):
            current = cand
        if dominance_key(cand) < dominance_key(best):
            best = cand
        temperature *= 0.94
    return best, current, generated


@dataclass
class MetaheuristicConfig:
    population_size: int
    eval_budget: int
    seed: int
    walltime_cap_s: float | None = None
    use_seed: bool = True
    use_jde: bool = True
    use_lns: bool = True
    lns_period: int = 10
    repair_budget: int = 12
    local_search_moves: int = 24
    diversity_restart: bool = True
    fixed_F: float = 0.72
    fixed_CR: float = 0.88
    deep_intensify: bool = False
    deep_polish_moves: int = 0
    use_trajectory_search: bool = False
    trajectory_time_fraction: float | None = None
    use_route_alns_endgame: bool = False
    route_endgame_reserve_s: float | None = None
    route_endgame_burst_iters: int | None = None


def default_population_size(customer_count: int) -> int:
    if customer_count <= 100:
        return 28
    if customer_count <= 200:
        return 36
    return 44


def ede_population_size(cfg: MetaheuristicConfig) -> int:
    # Leave enough budget for actual evolution instead of spending everything on initialization.
    if cfg.eval_budget <= 6:
        return min(cfg.population_size, 4)
    target = max(4, min(12, cfg.eval_budget // 3))
    if cfg.walltime_cap_s is not None and cfg.walltime_cap_s > 0:
        if cfg.walltime_cap_s <= 30:
            target = min(target, 6)
        elif cfg.walltime_cap_s <= 60:
            target = min(target, 8)
    return min(cfg.population_size, target)


def initialize_population(
    problem: ProblemInstance,
    references: tuple[float, float, float],
    cfg: MetaheuristicConfig,
    deadline: float | None = None,
) -> list[tuple[np.ndarray, Solution, float, float]]:
    rng = random.Random(cfg.seed)
    n = problem.customer_count
    target_population = ede_population_size(cfg)
    local_moves = _scaled_search_moves(cfg.local_search_moves, cfg.walltime_cap_s, minimum=6)
    deep_moves = _scaled_search_moves(cfg.deep_polish_moves, cfg.walltime_cap_s, minimum=4) if cfg.deep_polish_moves > 0 else 0
    if cfg.deep_intensify:
        reserve = max(2, cfg.eval_budget // 3) if cfg.use_trajectory_search else 2
        target_population = min(cfg.population_size, max(target_population, min(max(5, cfg.eval_budget - reserve), 9)))
    if cfg.walltime_cap_s is not None and cfg.walltime_cap_s > 0:
        if cfg.walltime_cap_s <= 30:
            target_population = min(target_population, 6)
        elif cfg.walltime_cap_s <= 60:
            target_population = min(target_population, 8)
    population: list[tuple[np.ndarray, Solution, float, float]] = []
    if cfg.use_seed:
        seed_perm = seed_permutation(problem, rng)
        keys = keys_from_permutation(seed_perm, rng)
        sol = evaluate_solution(problem, decode_permutation(problem, seed_perm, references), references)
        sol = bounded_repair(problem, sol, references, SearchStats(), max_attempts=cfg.repair_budget, deadline=deadline)
        sol = try_improve_with_local_search(problem, sol, references, max_moves=local_moves, deadline=deadline)
        if deep_moves > 0:
            sol = deep_route_polish(problem, sol, references, max_moves=deep_moves, deadline=deadline)
        if cfg.deep_intensify and not _time_exceeded(deadline):
            seed_walk_iters = 2 if cfg.walltime_cap_s is not None and cfg.walltime_cap_s <= 60 else 3
            seed_best, _, _ = incumbent_ils_burst(
                problem,
                sol,
                references,
                rng,
                SearchStats(),
                iterations=seed_walk_iters,
                repair_budget=cfg.repair_budget,
                ls_moves=max(10, local_moves),
                polish_moves=max(0, deep_moves // 2),
                deadline=deadline,
            )
            if dominance_key(seed_best) < dominance_key(sol):
                sol = seed_best
        population.append((keys, sol, cfg.fixed_F, cfg.fixed_CR))
        anchor_sol = sol
        anchored_variants = target_population - 1 if cfg.deep_intensify else min(target_population - 1, 4 if problem.customer_count <= 120 else 6)
        if cfg.walltime_cap_s is not None and cfg.walltime_cap_s > 0:
            if cfg.walltime_cap_s <= 30:
                anchored_variants = min(anchored_variants, 2)
            elif cfg.walltime_cap_s <= 60:
                anchored_variants = min(anchored_variants, 3)
            elif cfg.walltime_cap_s <= 300:
                anchored_variants = min(anchored_variants, 4)
        for _ in range(max(0, anchored_variants)):
            if _time_exceeded(deadline):
                return population
            pert_perm = perturbed_permutation_from_solution(problem, anchor_sol, rng)
            pert_keys = keys_from_permutation(pert_perm, rng)
            pert_sol = evaluate_solution(problem, decode_permutation(problem, pert_perm, references), references)
            pert_sol = bounded_repair(problem, pert_sol, references, SearchStats(), max_attempts=max(4, cfg.repair_budget // 2), deadline=deadline)
            pert_sol = try_improve_with_local_search(problem, pert_sol, references, max_moves=local_moves, deadline=deadline)
            if deep_moves > 0:
                pert_sol = deep_route_polish(problem, pert_sol, references, max_moves=deep_moves, deadline=deadline)
            population.append((pert_keys, pert_sol, cfg.fixed_F, cfg.fixed_CR))
            if dominance_key(pert_sol) < dominance_key(anchor_sol):
                anchor_sol = pert_sol
    while len(population) < target_population and not _time_exceeded(deadline):
        perm = list(problem.customer_ids)
        rng.shuffle(perm)
        if population and rng.random() < 0.55:
            anchor = permutation_from_solution(population[0][1])
            m = max(1, int(0.08 * n))
            for _ in range(m):
                i, j = rng.randrange(n), rng.randrange(n)
                anchor[i], anchor[j] = anchor[j], anchor[i]
            perm = anchor
        keys = keys_from_permutation(perm, rng)
        sol = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
        sol = bounded_repair(problem, sol, references, SearchStats(), max_attempts=max(2, cfg.repair_budget // 2), deadline=deadline)
        population.append((keys, sol, cfg.fixed_F, cfg.fixed_CR))
    return population


def jde_evolve(problem: ProblemInstance, cfg: MetaheuristicConfig, source_tag: str = "EDE") -> tuple[Solution, SearchStats, list[Solution]]:
    rng = random.Random(cfg.seed)
    deadline = _deadline_from_cap(cfg.walltime_cap_s)
    start = time.perf_counter()
    init_deadline = _phase_deadline(start, deadline, cfg.walltime_cap_s, 0.10)
    trajectory_deadline = _phase_deadline(start, deadline, cfg.walltime_cap_s, cfg.trajectory_time_fraction or 0.30)
    main_deadline = deadline
    endgame_reserve_s = 0.0
    if cfg.use_route_alns_endgame and cfg.walltime_cap_s is not None and 30 < cfg.walltime_cap_s <= 120:
        endgame_reserve_s = (
            cfg.route_endgame_reserve_s
            if cfg.route_endgame_reserve_s is not None
            else 6.0 if problem.customer_count <= 120 else 8.0 if problem.customer_count <= 240 else 10.0
        )
    evolve_local_moves = _scaled_search_moves(cfg.local_search_moves, cfg.walltime_cap_s, minimum=6)
    lns_moves = max(8, evolve_local_moves // 2)
    shake_moves = max(10, evolve_local_moves)
    trajectory_moves = max(12, evolve_local_moves)
    trajectory_polish = _scaled_search_moves(cfg.deep_polish_moves, cfg.walltime_cap_s, minimum=4) if cfg.deep_polish_moves > 0 else 0
    references = build_seed_references(problem)
    stats = SearchStats()
    population = initialize_population(problem, references, cfg, deadline=init_deadline)
    if not population:
        perm = list(problem.customer_ids)
        rng.shuffle(perm)
        keys = keys_from_permutation(perm, rng)
        sol = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
        population = [(keys, sol, cfg.fixed_F, cfg.fixed_CR)]
    archive: list[Solution] = []
    best = min((p[1] for p in population), key=dominance_key)
    init_best = best.score
    evals = len(population)
    for idx, (_, sol, _, _) in enumerate(population, start=1):
        archive = non_dominated_insert(archive, sol)
        if sol.accepted and stats.first_feasible_eval is None:
            stats.first_feasible_eval = idx
            stats.first_feasible_sec = time.perf_counter() - start
    if cfg.use_trajectory_search and evals < cfg.eval_budget and not _time_exceeded(trajectory_deadline):
        if cfg.walltime_cap_s is not None and cfg.walltime_cap_s > 0 and cfg.walltime_cap_s <= 30:
            traj_cap = 8 if problem.customer_count <= 120 else 12 if problem.customer_count <= 240 else 16
            traj_iters = min(max(4, traj_cap), cfg.eval_budget - evals)
        else:
            traj_iters = min(max(2, cfg.eval_budget // 4), cfg.eval_budget - evals)
        t_best, _, generated = trajectory_intensification(
            problem,
            best,
            references,
            rng,
            stats,
            iterations=traj_iters,
            repair_budget=cfg.repair_budget,
            ls_moves=trajectory_moves,
            polish_moves=trajectory_polish,
            deadline=trajectory_deadline,
        )
        for cand in generated:
            evals += 1
            archive = non_dominated_insert(archive, cand)
            if cand.accepted and stats.first_feasible_eval is None:
                stats.first_feasible_eval = evals
                stats.first_feasible_sec = time.perf_counter() - start
        if dominance_key(t_best) < dominance_key(best):
            best = t_best
            best_perm = permutation_from_solution(best)
            worst = max(range(len(population)), key=lambda idx: dominance_key(population[idx][1]))
            population[worst] = (keys_from_permutation(best_perm, rng), best, cfg.fixed_F, cfg.fixed_CR)
    gen = 0
    stagnation_gens = 0
    if len(population) < 4:
        stats.note("population_truncated=1")
    while evals < cfg.eval_budget and not _time_exceeded(main_deadline):
        if endgame_reserve_s > 0.0 and _remaining_time(deadline) <= endgame_reserve_s:
            break
        if len(population) < 4:
            break
        gen += 1
        improved_this_gen = False
        pop_scores = [p[1].score for p in population]
        div = float(np.std(np.array([k[:, 0] for k, _, _, _ in population]), axis=0).mean())
        best_keys = keys_from_permutation(permutation_from_solution(best), rng)
        for i in range(len(population)):
            if evals >= cfg.eval_budget:
                break
            if _time_exceeded(main_deadline):
                break
            if endgame_reserve_s > 0.0 and _remaining_time(deadline) <= endgame_reserve_s:
                break
            idxs = list(range(len(population)))
            idxs.remove(i)
            r1, r2, r3 = rng.sample(idxs, 3)
            target_keys, target_sol, target_F, target_CR = population[i]
            F, CR = target_F, target_CR
            if cfg.use_jde:
                if rng.random() < 0.1:
                    F = rng.uniform(0.5, 0.95)
                if rng.random() < 0.1:
                    CR = rng.uniform(0.6, 1.0)
                # diversity feedback
                F = min(0.95, max(0.5, F * (1.0 + 0.35 * (0.10 - div))))
                CR = min(1.0, max(0.55, CR * (1.0 - 0.25 * (0.10 - div))))
            if cfg.use_jde:
                donor = target_keys + 0.35 * (best_keys - target_keys) + F * (population[r1][0] - population[r2][0])
            else:
                donor = population[r1][0] + F * (population[r2][0] - population[r3][0])
            donor = np.mod(donor, 1.0)
            trial = target_keys.copy()
            j_rand = rng.randrange(problem.customer_count)
            for j in range(problem.customer_count):
                if rng.random() < CR or j == j_rand:
                    trial[j, :] = donor[j, :]
            perm = permutation_from_keys(trial)
            cand = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
            if not cand.accepted:
                cand = bounded_repair(problem, cand, references, stats, max_attempts=cfg.repair_budget, deadline=main_deadline)
            cand = try_improve_with_local_search(problem, cand, references, max_moves=evolve_local_moves, deadline=main_deadline)
            evals += 1
            if cand.accepted and stats.first_feasible_eval is None:
                stats.first_feasible_eval = evals
                stats.first_feasible_sec = time.perf_counter() - start
            archive = non_dominated_insert(archive, cand)
            if dominance_key(cand) <= dominance_key(target_sol):
                population[i] = (trial, cand, F, CR)
                if dominance_key(cand) < dominance_key(best):
                    best = cand
                    improved_this_gen = True
            else:
                stats.n_rejected_offspring += 1
        if (
            cfg.use_lns
            and gen % max(1, cfg.lns_period) == 0
            and evals < cfg.eval_budget
            and not _time_exceeded(main_deadline)
            and not (endgame_reserve_s > 0.0 and _remaining_time(deadline) <= endgame_reserve_s)
        ):
            lns_sol = boundary_lns(problem, best, references, rng, destroy_frac=0.16 if problem.customer_count <= 120 else 0.12, ls_moves=lns_moves, deadline=main_deadline)
            evals += 1
            archive = non_dominated_insert(archive, lns_sol)
            if dominance_key(lns_sol) < dominance_key(best):
                best = lns_sol
                improved_this_gen = True
        if (
            cfg.use_jde
            and cfg.use_lns
            and stagnation_gens >= 2
            and evals < cfg.eval_budget
            and not _time_exceeded(main_deadline)
            and not (endgame_reserve_s > 0.0 and _remaining_time(deadline) <= endgame_reserve_s)
        ):
            if cfg.use_route_alns_endgame and best.accepted:
                shaken_best, _, generated = route_alns_endgame(
                    problem,
                    best,
                    references,
                    rng,
                    stats,
                    iterations=1,
                    repair_budget=cfg.repair_budget,
                    ls_moves=max(12, shake_moves),
                    polish_moves=max(0, trajectory_polish // 2),
                    deadline=main_deadline,
                )
                for cand in generated:
                    evals += 1
                    archive = non_dominated_insert(archive, cand)
                shaken = shaken_best
                if generated:
                    stats.note("ede_route_shake=1")
            else:
                shaken = elite_route_perturbation(
                    problem,
                    best,
                    references,
                    rng,
                    stats,
                    repair_budget=cfg.repair_budget,
                    ls_moves=shake_moves,
                    polish_moves=trajectory_polish,
                    deadline=main_deadline,
                )
                evals += 1
                archive = non_dominated_insert(archive, shaken)
            if dominance_key(shaken) < dominance_key(best):
                best = shaken
                improved_this_gen = True
                shaken_perm = permutation_from_solution(shaken)
                worst = max(range(len(population)), key=lambda idx: dominance_key(population[idx][1]))
                population[worst] = (keys_from_permutation(shaken_perm, rng), shaken, cfg.fixed_F, cfg.fixed_CR)
        if (
            cfg.diversity_restart
            and gen % 8 == 0
            and div < 0.03
            and evals < cfg.eval_budget
            and not _time_exceeded(main_deadline)
            and not (endgame_reserve_s > 0.0 and _remaining_time(deadline) <= endgame_reserve_s)
        ):
            replace = max(1, len(population) // 6)
            for pos in sorted(range(len(population)), key=lambda idx: dominance_key(population[idx][1]), reverse=True)[:replace]:
                if _time_exceeded(main_deadline):
                    break
                perm = permutation_from_solution(best)
                for _ in range(max(2, problem.customer_count // 25)):
                    a, b = rng.randrange(problem.customer_count), rng.randrange(problem.customer_count)
                    perm[a], perm[b] = perm[b], perm[a]
                keys = keys_from_permutation(perm, rng)
                sol = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
                evals += 1
                population[pos] = (keys, sol, cfg.fixed_F, cfg.fixed_CR)
        stagnation_gens = 0 if improved_this_gen else stagnation_gens + 1
    if (
        cfg.use_jde
        and cfg.use_route_alns_endgame
        and cfg.walltime_cap_s is not None
        and 30 < cfg.walltime_cap_s <= 120
        and evals < cfg.eval_budget
        and not _time_exceeded(deadline)
        and (endgame_reserve_s <= 0.0 or _remaining_time(deadline) > 0.5)
    ):
        burst_iters = min(
            cfg.route_endgame_burst_iters if cfg.route_endgame_burst_iters is not None else 3 if problem.customer_count <= 120 else 2,
            cfg.eval_budget - evals,
        )
        b_best, _, generated = route_alns_endgame(
            problem,
            best,
            references,
            rng,
            stats,
            iterations=burst_iters,
            repair_budget=cfg.repair_budget,
            ls_moves=max(10, evolve_local_moves),
            polish_moves=max(0, trajectory_polish),
            deadline=deadline,
        )
        for cand in generated:
            evals += 1
            archive = non_dominated_insert(archive, cand)
            if cand.accepted and stats.first_feasible_eval is None:
                stats.first_feasible_eval = evals
                stats.first_feasible_sec = time.perf_counter() - start
        if dominance_key(b_best) < dominance_key(best):
            best = b_best
        stats.note(f"ede_route_endgame_iters={len(generated)}")
    best.source = source_tag
    stats.eval_count = evals
    stats.archive_size_final = len(archive)
    if _time_exceeded(deadline):
        stats.note("walltime_hit=1")
    stats.note(f"init_best={init_best:.6f}")
    return best, stats, archive


def alns_search(problem: ProblemInstance, cfg: MetaheuristicConfig) -> tuple[Solution, SearchStats, list[Solution]]:
    start = time.perf_counter()
    deadline = _deadline_from_cap(cfg.walltime_cap_s)
    references = build_seed_references(problem)
    stats = SearchStats()
    rng = random.Random(cfg.seed)
    seed_sol = evaluate_solution(problem, decode_permutation(problem, seed_permutation(problem, rng), references), references)
    current = bounded_repair(problem, seed_sol, references, stats, max_attempts=cfg.repair_budget, deadline=deadline)
    current = try_improve_with_local_search(problem, current, references, max_moves=cfg.local_search_moves, deadline=deadline)
    best = current
    archive = non_dominated_insert([], best)
    destroy_names = ["random", "worst", "related", "border"]
    weights = {name: 1.0 for name in destroy_names}
    evals = 1
    if best.accepted:
        stats.first_feasible_eval = evals
        stats.first_feasible_sec = time.perf_counter() - start
    while evals < cfg.eval_budget and not _time_exceeded(deadline):
        names, probs = zip(*[(n, w / sum(weights.values())) for n, w in weights.items()])
        op = rng.choices(names, weights=probs, k=1)[0]
        removed_count = max(2, int(problem.customer_count * (0.06 if problem.customer_count <= 120 else 0.04)))
        perm = permutation_from_solution(current)
        if op == "random":
            removed = set(rng.sample(perm, min(removed_count, len(perm))))
        elif op == "worst":
            contrib = []
            for cid in perm:
                if _time_exceeded(deadline):
                    break
                temp = [x for x in perm if x != cid]
                sol = evaluate_solution(problem, decode_permutation(problem, temp, references), references)
                evals += 1
                contrib.append((current.score - sol.score, cid))
                if evals >= cfg.eval_budget:
                    break
            contrib.sort(reverse=True)
            removed = {cid for _, cid in contrib[:removed_count]}
        elif op == "related":
            anchor = rng.choice(perm)
            ax, ay = customer_pos(problem, anchor)
            near = sorted(perm, key=lambda cid: euclidean((ax, ay), customer_pos(problem, cid)))
            removed = set(near[:removed_count])
        else:
            border = route_boundary_customers(problem, current)
            base = border if len(border) >= removed_count else perm
            removed = set(base[:removed_count])
        base_perm = [cid for cid in perm if cid not in removed]
        partial = evaluate_solution(problem, decode_permutation(problem, base_perm, references), references)
        cand = partial
        for cid in sorted(removed, key=lambda c: problem.customers[c].tw_start):
            inserts = _candidate_insertions(problem, cand.routes, cid, references)
            inserts.sort(key=dominance_key)
            cand = inserts[0]
            evals += 1
            if evals >= cfg.eval_budget or _time_exceeded(deadline):
                break
        cand = bounded_repair(problem, cand, references, stats, max_attempts=cfg.repair_budget, deadline=deadline)
        cand = try_improve_with_local_search(problem, cand, references, max_moves=cfg.local_search_moves, deadline=deadline)
        archive = non_dominated_insert(archive, cand)
        if dominance_key(cand) < dominance_key(best):
            best = cand
            current = cand
            weights[op] += 2.0
        elif dominance_key(cand) <= dominance_key(current) or rng.random() < 0.08:
            current = cand
            weights[op] += 0.5
        else:
            stats.n_rejected_offspring += 1
            weights[op] = max(0.2, weights[op] * 0.95)
        evals += 1
        if cand.accepted and stats.first_feasible_eval is None:
            stats.first_feasible_eval = evals
            stats.first_feasible_sec = time.perf_counter() - start
    stats.eval_count = evals
    stats.archive_size_final = len(archive)
    if _time_exceeded(deadline):
        stats.note("walltime_hit=1")
    return best, stats, archive


def order_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n  # type: ignore[list-item]
    child[a:b] = p1[a:b]
    fill = [x for x in p2 if x not in child]
    ptr = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[ptr]
            ptr += 1
    return [int(x) for x in child]


def hgs_search(problem: ProblemInstance, cfg: MetaheuristicConfig) -> tuple[Solution, SearchStats, list[Solution]]:
    rng = random.Random(cfg.seed)
    deadline = _deadline_from_cap(cfg.walltime_cap_s)
    start = time.perf_counter()
    references = build_seed_references(problem)
    stats = SearchStats()
    pop_size = min(cfg.population_size, max(8, min(cfg.eval_budget, 20)))
    population: list[tuple[list[int], Solution]] = []
    seed_perm = seed_permutation(problem, rng)
    seed_sol = evaluate_solution(problem, decode_permutation(problem, seed_perm, references), references)
    population.append((seed_perm, try_improve_with_local_search(problem, seed_sol, references, max_moves=cfg.local_search_moves, deadline=deadline)))
    while len(population) < pop_size and not _time_exceeded(deadline):
        perm = list(problem.customer_ids)
        rng.shuffle(perm)
        sol = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
        population.append((perm, sol))
    best = min((s for _, s in population), key=dominance_key)
    archive = []
    for _, sol in population:
        archive = non_dominated_insert(archive, sol)
    evals = len(population)
    while evals < cfg.eval_budget and not _time_exceeded(deadline):
        parents = rng.sample(population, 2)
        child_perm = order_crossover(parents[0][0], parents[1][0], rng)
        for _ in range(max(1, problem.customer_count // 40)):
            a, b = rng.sample(range(problem.customer_count), 2)
            child_perm[a], child_perm[b] = child_perm[b], child_perm[a]
        child_sol = evaluate_solution(problem, decode_permutation(problem, child_perm, references), references)
        child_sol = bounded_repair(problem, child_sol, references, stats, max_attempts=cfg.repair_budget, deadline=deadline)
        child_sol = try_improve_with_local_search(problem, child_sol, references, max_moves=cfg.local_search_moves, deadline=deadline)
        evals += 1
        archive = non_dominated_insert(archive, child_sol)
        population.append((child_perm, child_sol))
        population.sort(key=lambda item: (dominance_key(item[1]), diversity_penalty(item[0], population)))
        population = population[:pop_size]
        if dominance_key(child_sol) < dominance_key(best):
            best = child_sol
    stats.eval_count = evals
    stats.archive_size_final = len(archive)
    if _time_exceeded(deadline):
        stats.note("walltime_hit=1")
    return best, stats, archive


def diversity_penalty(perm: list[int], population: list[tuple[list[int], Solution]]) -> float:
    if len(population) <= 1:
        return 0.0
    n = len(perm)
    sample = population[: min(5, len(population))]
    dists = []
    for other, _ in sample:
        d = sum(1 for a, b in zip(perm, other) if a != b) / max(1, n)
        dists.append(d)
    return -float(np.mean(dists))


def ils_search(problem: ProblemInstance, cfg: MetaheuristicConfig) -> tuple[Solution, SearchStats, list[Solution]]:
    rng = random.Random(cfg.seed)
    deadline = _deadline_from_cap(cfg.walltime_cap_s)
    start = time.perf_counter()
    references = build_seed_references(problem)
    stats = SearchStats()
    perm = seed_permutation(problem, rng)
    current = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
    current = bounded_repair(problem, current, references, stats, max_attempts=cfg.repair_budget, deadline=deadline)
    current = try_improve_with_local_search(problem, current, references, max_moves=cfg.local_search_moves * 2, deadline=deadline)
    best = current
    archive = non_dominated_insert([], best)
    evals = 1
    temperature = 0.05
    while evals < cfg.eval_budget and not _time_exceeded(deadline):
        base_perm = permutation_from_solution(current)
        # shift-border perturbation
        border = route_boundary_customers(problem, current)
        if len(border) >= 2:
            a, b = rng.sample(border, 2)
            ia, ib = base_perm.index(a), base_perm.index(b)
            base_perm[ia], base_perm[ib] = base_perm[ib], base_perm[ia]
        else:
            for _ in range(max(2, problem.customer_count // 35)):
                i, j = rng.sample(range(problem.customer_count), 2)
                base_perm[i], base_perm[j] = base_perm[j], base_perm[i]
        cand = evaluate_solution(problem, decode_permutation(problem, base_perm, references), references)
        cand = bounded_repair(problem, cand, references, stats, max_attempts=cfg.repair_budget, deadline=deadline)
        cand = try_improve_with_local_search(problem, cand, references, max_moves=cfg.local_search_moves, deadline=deadline)
        evals += 1
        archive = non_dominated_insert(archive, cand)
        delta = cand.score - current.score
        if dominance_key(cand) < dominance_key(current) or rng.random() < math.exp(-max(0.0, delta) / max(temperature, 1e-6)):
            current = cand
        else:
            stats.n_rejected_offspring += 1
        if dominance_key(cand) < dominance_key(best):
            best = cand
        temperature *= 0.996
    stats.eval_count = evals
    stats.archive_size_final = len(archive)
    if _time_exceeded(deadline):
        stats.note("walltime_hit=1")
    return best, stats, archive
