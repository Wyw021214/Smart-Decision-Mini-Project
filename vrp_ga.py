import numpy as np
import random, time, math
from typing import List, Dict, Any, Union, Tuple
from utils import split_tour_into_routes, vrp_routes_cost

# ---------------- Heterogeneous decoder ----------------
def _decode_hetero(order: List[int],
                   demands: List[int],
                   vehicle_caps: List[int]) -> Tuple[List[List[int]], List[int], int]:
    """
    Decode a permutation of customers into routes for a heterogeneous fleet.

    - order: list of customers indices (1..n)
    - demands: demand per node (demands[0] = 0 for depot)
    - vehicle_caps: list of capacities for each vehicle

    Returns:
      routes: one list per vehicle (customers served by that vehicle)
      loads: realized load per vehicle
      used_k: number of vehicles actually used (non-empty routes)
    """
    k = len(vehicle_caps)
    routes = [[] for _ in range(k)]
    loads  = [0  for _ in range(k)]

    v = 0
    for c in order:
        d = int(demands[c])
        # If not the last vehicle, try to respect capacity.
        if v < k - 1 and loads[v] + d > int(vehicle_caps[v]):
            v += 1
        routes[v].append(c)
        loads[v] += d

    used_k = sum(1 for r in routes if r)
    return routes, loads, used_k

def _hetero_cost(routes: List[List[int]],
                 loads: List[int],
                 vehicle_caps: List[int],
                 D: np.ndarray,
                 penalty_overflow: float = 1e6) -> Tuple[float, int]:
    """
    Compute distance + penalty for heterogeneous fleet routes.

    - routes: list of customer routes (per vehicle)
    - loads: realized loads per vehicle
    - vehicle_caps: capacities per vehicle
    - D: distance matrix
    - penalty_overflow: penalty per unit of overflow

    Returns:
      (cost_with_penalty, vehicles_used)
    """
    dist = vrp_routes_cost(routes, D)
    overflow = sum(max(0, loads[i] - int(vehicle_caps[i])) for i in range(len(vehicle_caps)))
    penalty = penalty_overflow * overflow
    used_k = sum(1 for r in routes if r)
    return dist + penalty, used_k

# ---------------- Objective helpers ----------------
def _lexi_cost(vehicles_used: int, distance: float, big_m: float = 1e6) -> float:
    """
    Lexicographic scalarization: prioritize number of vehicles, then distance.
    BIG_M ensures that any difference in vehicle count dominates distance difference.
    """
    return vehicles_used * big_m + distance

# ====================================================
#   GA supporting homogeneous & heterogeneous fleets
# ====================================================
def run_vrp_ga(D: np.ndarray,
               demands: List[int],
               vehicle_capacities: Union[int, List[int]],
               objective: str = "min_vehicles_then_distance",
               max_vehicles: int = 999,
               penalty_overflow: float = 1e6,
               n_pop: int = 200,
               iters: int = 1000,
               cx_rate: float = 0.8,
               mut_rate: float = 0.2,
               seed: int = 0) -> Dict[str, Any]:
    """
    Genetic Algorithm for the Vehicle Routing Problem.

    Supports both:
      - Homogeneous fleet: vehicle_capacities is an int (all vehicles same capacity).
      - Heterogeneous fleet: vehicle_capacities is a list of capacities per vehicle.

    Objective:
      - "min_vehicles_then_distance": minimize number of vehicles first, then distance.
      - "min_distance": minimize distance only.

    Parameters
    ----------
    D : np.ndarray
        (n+1)x(n+1) distance matrix (0 is depot).
    demands : List[int]
        demands[i] is demand of customer i (demands[0] = 0).
    vehicle_capacities : int | List[int]
        Homogeneous capacity (int) or list of heterogeneous capacities.
    max_vehicles : int
        Only used for homogeneous case to cap the fleet size.
    penalty_overflow : float
        Penalty coefficient for capacity overflow (heterogeneous decoding).
    n_pop, iters, cx_rate, mut_rate, seed : GA hyperparameters.

    Returns
    -------
    dict with keys:
        - routes: list of routes (each route = list of customers, depot excluded)
        - best_cost: best scalarized cost (depends on objective)
        - history: [(iteration, best_cost_so_far)]
        - runtime: execution time in seconds
        - vehicles_used: number of vehicles actually used
        - vehicle_capacities: list of capacities (for reference)
        - loads (only for heterogeneous): realized load per vehicle
        - chromosome: best permutation of customers
        - objective: objective string
    """
    rng = random.Random(seed)
    n = D.shape[0] - 1  # number of customers

    # Normalize fleet definition
    is_homogeneous = isinstance(vehicle_capacities, int)
    if is_homogeneous:
        cap = int(vehicle_capacities)
        total_demand = int(sum(demands[1:]))
        est_needed = max(1, math.ceil(total_demand / max(1, cap)))
        k_hint = est_needed if max_vehicles is None else max_vehicles
        fleet_caps = [cap] * int(k_hint)
    else:
        fleet_caps = [int(x) for x in vehicle_capacities]
        if len(fleet_caps) < 1:
            raise ValueError("vehicle_capacities list must be non-empty for heterogeneous fleet.")

    BIG_M = 1e6  # large constant for lexicographic objective

    def decode_cost(order: List[int]) -> float:
        if is_homogeneous:
            # Homogeneous: greedy split by capacity
            routes = split_tour_into_routes(order, demands, cap)
            dist = vrp_routes_cost(routes, D)
            vehicles_used = len(routes)
            return dist if objective == "min_distance" else _lexi_cost(vehicles_used, dist, BIG_M)
        else:
            # Heterogeneous: sequential allocation by capacities
            routes, loads, used_k = _decode_hetero(order, demands, fleet_caps)
            dist_pen, used_k2 = _hetero_cost(routes, loads, fleet_caps, D, penalty_overflow)
            return dist_pen if objective == "min_distance" else _lexi_cost(used_k2, dist_pen, BIG_M)

    # ----- GA operators -----
    def init_pop():
        base = list(range(1, n+1))
        pop = []
        for _ in range(n_pop):
            rng.shuffle(base)
            pop.append(base[:])
        return pop

    def ordered_crossover(p1, p2):
        a, b = sorted(rng.sample(range(n), 2))
        hole = set(p1[a:b+1])
        child = [None]*n
        child[a:b+1] = p1[a:b+1]
        fill = [x for x in p2 if x not in hole]
        j = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[j]; j += 1
        return child

    def mutate_swap(ch):
        i, j = rng.sample(range(n), 2); ch[i], ch[j] = ch[j], ch[i]

    # ----- GA loop -----
    pop = init_pop()
    fitness = [1.0 / (decode_cost(ind) + 1e-9) for ind in pop]
    best = pop[int(np.argmax(fitness))][:]
    best_cost = 1.0 / max(fitness)
    history = [(0, best_cost)]
    t0 = time.time()

    for it in range(1, iters+1):
        new_pop = [best[:]]
        while len(new_pop) < n_pop:
            a, b = rng.sample(range(n_pop), 2)
            c, d = rng.sample(range(n_pop), 2)
            p1 = pop[a] if fitness[a] > fitness[b] else pop[b]
            p2 = pop[c] if fitness[c] > fitness[d] else pop[d]
            child = ordered_crossover(p1, p2) if rng.random() < cx_rate else p1[:]
            if rng.random() < mut_rate: mutate_swap(child)
            new_pop.append(child)
        pop = new_pop
        fitness = [1.0 / (decode_cost(ind) + 1e-9) for ind in pop]
        cur_best_idx = int(np.argmax(fitness))
        cur_best = pop[cur_best_idx][:]
        cur_best_cost = 1.0 / fitness[cur_best_idx]
        if cur_best_cost < best_cost:
            best_cost = cur_best_cost; best = cur_best[:]
        history.append((it, best_cost))

    runtime = time.time() - t0

    # ----- Decode best chromosome -----
    if is_homogeneous:
        routes = split_tour_into_routes(best, demands, cap)
        dist = vrp_routes_cost(routes, D)
        vehicles_used = len(routes)
        out_best_cost = dist if objective == "min_distance" else _lexi_cost(vehicles_used, dist, BIG_M)
        return {
            "routes": routes,
            "best_cost": out_best_cost,
            "history": history,
            "runtime": runtime,
            "vehicles_used": vehicles_used,
            "vehicle_capacities": fleet_caps,
            "chromosome": best,
            "objective": objective,
        }
    else:
        routes, loads, used_k = _decode_hetero(best, demands, fleet_caps)
        dist_pen, used_k2 = _hetero_cost(routes, loads, fleet_caps, D, penalty_overflow)
        out_best_cost = dist_pen if objective == "min_distance" else _lexi_cost(used_k2, dist_pen, BIG_M)
        return {
            "routes": routes,
            "loads": loads,
            "vehicles_used": used_k2,
            "vehicle_capacities": fleet_caps,
            "best_cost": out_best_cost,
            "history": history,
            "runtime": runtime,
            "chromosome": best,
            "objective": objective,
        }
