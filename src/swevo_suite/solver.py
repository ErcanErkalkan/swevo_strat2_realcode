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
    shift = next(s for s in problem.shifts if s.idx == route.shift_id)
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


def _candidate_insertions(problem: ProblemInstance, routes: Sequence[Route], cid: int, references: tuple[float, float, float]) -> List[Solution]:
    proposals: List[Solution] = []
    vehicle_names = ["HEV", "EV", "ICE"]
    cust = problem.customers[cid]
    cust_mid = 0.5 * (cust.tw_start + cust.tw_end)

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
            new_routes = [Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in routes]
            new_routes[ridx].customers.insert(pos, cid)
            proposals.append(evaluate_solution(problem, new_routes, references))
    for shift in problem.shifts:
        for vehicle_name in vehicle_names[:2]:
            new_routes = [Route(r.shift_id, r.vehicle_type, list(r.customers)) for r in routes]
            new_routes.append(Route(shift.idx, vehicle_name, [cid]))
            proposals.append(evaluate_solution(problem, new_routes, references))
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
) -> Solution:
    best = sol.copy()
    moves = 0
    improved = True
    while improved and moves < max_moves:
        improved = False
        # relocate
        for ridx, route in enumerate(best.routes):
            if moves >= max_moves:
                break
            for pos, cid in enumerate(list(route.customers)):
                for tridx, troute in enumerate(best.routes):
                    for tpos in range(len(troute.customers) + 1):
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
) -> Solution:
    current = sol.copy()
    for _ in range(max_attempts):
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
        proposals = _candidate_insertions(problem, current.routes, cid, references)
        proposals.sort(key=dominance_key)
        current = proposals[0]
    current = try_improve_with_local_search(problem, current, references, max_moves=ls_moves)
    return current


@dataclass
class MetaheuristicConfig:
    population_size: int
    eval_budget: int
    seed: int
    use_seed: bool = True
    use_jde: bool = True
    use_lns: bool = True
    lns_period: int = 10
    repair_budget: int = 12
    local_search_moves: int = 24
    diversity_restart: bool = True
    fixed_F: float = 0.72
    fixed_CR: float = 0.88


def default_population_size(customer_count: int) -> int:
    if customer_count <= 100:
        return 28
    if customer_count <= 200:
        return 36
    return 44


def initialize_population(problem: ProblemInstance, references: tuple[float, float, float], cfg: MetaheuristicConfig) -> list[tuple[np.ndarray, Solution, float, float]]:
    rng = random.Random(cfg.seed)
    n = problem.customer_count
    target_population = min(cfg.population_size, max(6, min(cfg.eval_budget, 12)))
    population: list[tuple[np.ndarray, Solution, float, float]] = []
    if cfg.use_seed:
        seed_perm = seed_permutation(problem, rng)
        keys = keys_from_permutation(seed_perm, rng)
        sol = evaluate_solution(problem, decode_permutation(problem, seed_perm, references), references)
        sol = bounded_repair(problem, sol, references, SearchStats(), max_attempts=cfg.repair_budget)
        sol = try_improve_with_local_search(problem, sol, references, max_moves=cfg.local_search_moves)
        population.append((keys, sol, cfg.fixed_F, cfg.fixed_CR))
    while len(population) < target_population:
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
        sol = bounded_repair(problem, sol, references, SearchStats(), max_attempts=max(2, cfg.repair_budget // 2))
        population.append((keys, sol, cfg.fixed_F, cfg.fixed_CR))
    return population


def jde_evolve(problem: ProblemInstance, cfg: MetaheuristicConfig, source_tag: str = "EDE") -> tuple[Solution, SearchStats, list[Solution]]:
    rng = random.Random(cfg.seed)
    references = build_seed_references(problem)
    stats = SearchStats()
    start = time.perf_counter()
    population = initialize_population(problem, references, cfg)
    archive: list[Solution] = []
    best = min((p[1] for p in population), key=dominance_key)
    init_best = best.score
    evals = len(population)
    for _, sol, _, _ in population:
        archive = non_dominated_insert(archive, sol)
        if sol.accepted and stats.first_feasible_eval is None:
            stats.first_feasible_eval = evals
            stats.first_feasible_sec = time.perf_counter() - start
    gen = 0
    while evals < cfg.eval_budget:
        gen += 1
        pop_scores = [p[1].score for p in population]
        div = float(np.std(np.array([k[:, 0] for k, _, _, _ in population]), axis=0).mean())
        for i in range(len(population)):
            if evals >= cfg.eval_budget:
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
            donor = population[r1][0] + F * (population[r2][0] - population[r3][0])
            donor = np.mod(donor, 1.0)
            trial = target_keys.copy()
            j_rand = rng.randrange(problem.customer_count)
            for j in range(problem.customer_count):
                if rng.random() < CR or j == j_rand:
                    trial[j, :] = donor[j, :]
            perm = permutation_from_keys(trial)
            cand = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
            evals += 1
            if not cand.accepted:
                cand = bounded_repair(problem, cand, references, stats, max_attempts=cfg.repair_budget)
            cand = try_improve_with_local_search(problem, cand, references, max_moves=cfg.local_search_moves)
            evals += 1
            if cand.accepted and stats.first_feasible_eval is None:
                stats.first_feasible_eval = evals
                stats.first_feasible_sec = time.perf_counter() - start
            archive = non_dominated_insert(archive, cand)
            if dominance_key(cand) <= dominance_key(target_sol):
                population[i] = (trial, cand, F, CR)
                if dominance_key(cand) < dominance_key(best):
                    best = cand
            else:
                stats.n_rejected_offspring += 1
        if cfg.use_lns and gen % max(1, cfg.lns_period) == 0 and evals < cfg.eval_budget:
            lns_sol = boundary_lns(problem, best, references, rng, destroy_frac=0.16 if problem.customer_count <= 120 else 0.12, ls_moves=max(16, cfg.local_search_moves // 2))
            evals += 1
            archive = non_dominated_insert(archive, lns_sol)
            if dominance_key(lns_sol) < dominance_key(best):
                best = lns_sol
        if cfg.diversity_restart and gen % 8 == 0 and div < 0.03 and evals < cfg.eval_budget:
            replace = max(1, len(population) // 6)
            for pos in sorted(range(len(population)), key=lambda idx: dominance_key(population[idx][1]), reverse=True)[:replace]:
                perm = permutation_from_solution(best)
                for _ in range(max(2, problem.customer_count // 25)):
                    a, b = rng.randrange(problem.customer_count), rng.randrange(problem.customer_count)
                    perm[a], perm[b] = perm[b], perm[a]
                keys = keys_from_permutation(perm, rng)
                sol = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
                evals += 1
                population[pos] = (keys, sol, cfg.fixed_F, cfg.fixed_CR)
    best.source = source_tag
    stats.eval_count = evals
    stats.archive_size_final = len(archive)
    stats.note(f"init_best={init_best:.6f}")
    return best, stats, archive


def alns_search(problem: ProblemInstance, cfg: MetaheuristicConfig) -> tuple[Solution, SearchStats, list[Solution]]:
    references = build_seed_references(problem)
    stats = SearchStats()
    rng = random.Random(cfg.seed)
    start = time.perf_counter()
    seed_sol = evaluate_solution(problem, decode_permutation(problem, seed_permutation(problem, rng), references), references)
    current = bounded_repair(problem, seed_sol, references, stats, max_attempts=cfg.repair_budget)
    current = try_improve_with_local_search(problem, current, references, max_moves=cfg.local_search_moves)
    best = current
    archive = non_dominated_insert([], best)
    destroy_names = ["random", "worst", "related", "border"]
    weights = {name: 1.0 for name in destroy_names}
    evals = 1
    if best.accepted:
        stats.first_feasible_eval = evals
        stats.first_feasible_sec = time.perf_counter() - start
    while evals < cfg.eval_budget:
        names, probs = zip(*[(n, w / sum(weights.values())) for n, w in weights.items()])
        op = rng.choices(names, weights=probs, k=1)[0]
        removed_count = max(2, int(problem.customer_count * (0.06 if problem.customer_count <= 120 else 0.04)))
        perm = permutation_from_solution(current)
        if op == "random":
            removed = set(rng.sample(perm, min(removed_count, len(perm))))
        elif op == "worst":
            contrib = []
            for cid in perm:
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
            if evals >= cfg.eval_budget:
                break
        cand = bounded_repair(problem, cand, references, stats, max_attempts=cfg.repair_budget)
        cand = try_improve_with_local_search(problem, cand, references, max_moves=cfg.local_search_moves)
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
    references = build_seed_references(problem)
    stats = SearchStats()
    pop_size = min(cfg.population_size, max(8, min(cfg.eval_budget, 20)))
    population: list[tuple[list[int], Solution]] = []
    seed_perm = seed_permutation(problem, rng)
    seed_sol = evaluate_solution(problem, decode_permutation(problem, seed_perm, references), references)
    population.append((seed_perm, try_improve_with_local_search(problem, seed_sol, references, max_moves=cfg.local_search_moves)))
    while len(population) < pop_size:
        perm = list(problem.customer_ids)
        rng.shuffle(perm)
        sol = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
        population.append((perm, sol))
    best = min((s for _, s in population), key=dominance_key)
    archive = []
    for _, sol in population:
        archive = non_dominated_insert(archive, sol)
    evals = len(population)
    while evals < cfg.eval_budget:
        parents = rng.sample(population, 2)
        child_perm = order_crossover(parents[0][0], parents[1][0], rng)
        for _ in range(max(1, problem.customer_count // 40)):
            a, b = rng.sample(range(problem.customer_count), 2)
            child_perm[a], child_perm[b] = child_perm[b], child_perm[a]
        child_sol = evaluate_solution(problem, decode_permutation(problem, child_perm, references), references)
        child_sol = bounded_repair(problem, child_sol, references, stats, max_attempts=cfg.repair_budget)
        child_sol = try_improve_with_local_search(problem, child_sol, references, max_moves=cfg.local_search_moves)
        evals += 1
        archive = non_dominated_insert(archive, child_sol)
        population.append((child_perm, child_sol))
        population.sort(key=lambda item: (dominance_key(item[1]), diversity_penalty(item[0], population)))
        population = population[:pop_size]
        if dominance_key(child_sol) < dominance_key(best):
            best = child_sol
    stats.eval_count = evals
    stats.archive_size_final = len(archive)
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
    references = build_seed_references(problem)
    stats = SearchStats()
    perm = seed_permutation(problem, rng)
    current = evaluate_solution(problem, decode_permutation(problem, perm, references), references)
    current = bounded_repair(problem, current, references, stats, max_attempts=cfg.repair_budget)
    current = try_improve_with_local_search(problem, current, references, max_moves=cfg.local_search_moves * 2)
    best = current
    archive = non_dominated_insert([], best)
    evals = 1
    temperature = 0.05
    while evals < cfg.eval_budget:
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
        cand = bounded_repair(problem, cand, references, stats, max_attempts=cfg.repair_budget)
        cand = try_improve_with_local_search(problem, cand, references, max_moves=cfg.local_search_moves)
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
    return best, stats, archive
