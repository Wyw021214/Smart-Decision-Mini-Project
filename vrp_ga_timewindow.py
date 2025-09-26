"""
vrp_ga_timewindow.py
--------------------
Genetic Algorithm for VRPTW (Vehicle Routing Problem with Time Windows).
- Indirect encoding: chromosome = permutation of customers (1..n), depot is 0.
- Decoder builds capacity- and time-window-feasible routes (greedy), with waiting allowed.
- Travel time == Euclidean distance (speed = 1). Service times are added at each stop.

You can load Solomon-style instances (.txt) using `load_vrptw_from_txt(...)` and then run:
    coords, D, demands, tw, service, veh_cap, veh_num = load_vrptw_from_txt("C101.txt")
    res = run_vrp_ga_timewindow(D, demands, veh_cap, tw, service,
                                objective="min_vehicles_then_distance")

The parser expects a format like the uploaded file (VEHICLE section, then CUSTOMER table). :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Union
import math
import random
import time
import numpy as np


# ---------------- Utilities ----------------

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean(coords[i], coords[j])
            D[i, j] = D[j, i] = d
    return D


# ---------------- Parser for Solomon-like TXT ----------------

def load_vrptw_from_txt(path: str):
    """
    Parse a Solomon-style VRPTW .txt (like C101):
      - VEHICLE section with NUMBER and CAPACITY
      - CUSTOMER section with columns:
        CUST NO., XCOORD., YCOORD., DEMAND, READY TIME, DUE DATE, SERVICE TIME

    Returns:
      coords      : np.ndarray (n+1, 2)  (index 0 = depot)
      D           : np.ndarray pairwise Euclidean distances
      demands     : List[int] length n+1 (demands[0] = 0)
      time_windows: List[Tuple[float,float]] length n+1 (depot window kept)
      service     : List[float] length n+1 (service[0] typically 0)
      veh_cap     : int   (capacity per vehicle)
      veh_num     : int   (number of vehicles suggested by the file)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # read VEHICLE
    veh_cap, veh_num = None, None
    i = 0
    while i < len(lines):
        ln = lines[i].upper()
        if ln.startswith("VEHICLE"):
            # Next lines contain "NUMBER CAPACITY"
            # We scan forward to the first line that looks like two integers.
            j = i + 1
            while j < len(lines):
                parts = lines[j].split()
                ints = [p for p in parts if p.isdigit()]
                if len(ints) >= 2:
                    veh_num = int(ints[0])
                    veh_cap = int(ints[1])
                    break
                j += 1
            i = j
            break
        i += 1
    if veh_cap is None or veh_num is None:
        raise ValueError("Could not parse VEHICLE NUMBER/CAPACITY.")

    # read CUSTOMER rows
    # Find the line that starts with "CUSTOMER"
    cust_start = None
    for k, ln in enumerate(lines):
        if ln.upper().startswith("CUSTOMER"):
            cust_start = k
            break
    if cust_start is None:
        raise ValueError("Missing CUSTOMER section.")

    # The table typically starts 1–3 lines after this header; detect numeric rows
    rows = []
    for ln in lines[cust_start + 1 :]:
        # Try to parse a numeric row (at least 7 numeric tokens)
        parts = ln.split()
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                pass
        if len(nums) >= 7:
            # Expected order:
            # [cust_id, x, y, demand, ready, due, service]
            cid, x, y, d, r, due, svc = nums[:7]
            rows.append((int(cid), float(x), float(y), int(d), float(r), float(due), float(svc)))

    # sort by customer id to ensure 0..n order
    rows.sort(key=lambda t: t[0])
    n = len(rows) - 1  # customers (excluding depot 0)

    # build outputs
    coords = np.array([[r[1], r[2]] for r in rows], dtype=float)
    demands = [int(r[3]) for r in rows]
    time_windows = [(float(r[4]), float(r[5])) for r in rows]
    service = [float(r[6]) for r in rows]
    D = pairwise_distance_matrix(coords)

    return coords, D, demands, time_windows, service, int(veh_cap), int(veh_num)


# ---------------- Decoder with Time Windows ----------------

def split_tour_into_routes_timewindow(
    order: List[int],
    demands: List[int],
    capacity: int,
    time_windows: List[Tuple[float, float]],
    service_times: List[float],
    D: np.ndarray,
    late_penalty: float = 1e6,
) -> Tuple[List[List[int]], float]:
    """
    Greedy decoder for VRPTW with capacity and time windows.

    - order          : permutation of customers (1..n)
    - capacity       : homogeneous vehicle capacity
    - time_windows   : [(ready, due)] for nodes 0..n
    - service_times  : service time per node (0..n); service_times[0] often 0
    - D              : distance/time matrix
    - late_penalty   : penalty added per unit of tardiness if a customer is served after 'due'

    Decoding logic (per route):
      start at depot at t=0; for each customer c in 'order':
        * try to append to current route:
            arrival = current_time + travel
            begin   = max(arrival, ready[c])   # waiting if early
            if begin > due[c]:
                -> close current route (return to depot),
                   open a new route from depot at t=0 and assign c there.
            else:
                -> accept; time = begin + service[c]
           (Also enforce capacity; if adding c exceeds capacity, open a new route.)
      Each route returns to depot at the end.
      If any service begins after due[c], add tardiness penalty (should be rare with the above check).

    Returns:
      routes : list of routes (list of customers, depot excluded)
      cost   : total distance + tardiness penalty (if any)
    """
    routes: List[List[int]] = []
    cur: List[int] = []
    load = 0
    time_now = 0.0
    last = 0  # depot

    def flush_route():
        nonlocal routes, cur, load, time_now, last
        if cur:
            routes.append(cur)
        cur = []
        load = 0
        time_now = 0.0
        last = 0

    penalty = 0.0

    for c in order:
        dem = int(demands[c])
        ready_c, due_c = time_windows[c]
        svc_c = service_times[c]

        # If capacity would overflow -> start new vehicle/route
        if load + dem > capacity and cur:
            # close current route (return to depot)
            time_now += D[last, 0]
            flush_route()

        # check time feasibility from current state
        arrival = time_now + D[last, c]
        begin = max(arrival, ready_c)  # wait if early
        if begin > due_c and cur:
            # can't insert here due to window; close route and start new one
            time_now += D[last, 0]
            flush_route()
            # recompute from new route (depot at t=0)
            arrival = D[0, c]
            begin = max(arrival, ready_c)

        # accept c in the (new) current route
        cur.append(c)
        load += dem
        last = c
        # tardiness (should be zero if we already split above, but keep as safety)
        tardiness = max(0.0, begin - due_c)
        penalty += tardiness * late_penalty
        time_now = begin + svc_c

    # finish last route
    if cur:
        time_now += D[last, 0]
        routes.append(cur)

    # base distance
    dist = 0.0
    for r in routes:
        prev = 0
        for c in r:
            dist += D[prev, c]
            prev = c
        dist += D[prev, 0]

    return routes, dist + penalty


# ---------------- GA for VRPTW ----------------

def run_vrp_ga_timewindow(
    D: np.ndarray,
    demands: List[int],
    capacity: int,
    time_windows: List[Tuple[float, float]],
    service_times: List[float],
    n_pop: int = 200,
    iters: int = 1000,
    cx_rate: float = 0.8,
    mut_rate: float = 0.2,
    seed: int = 0,
    objective: str = "min_vehicles_then_distance",
) -> Dict[str, Any]:
    """
    GA wrapper for VRPTW (homogeneous capacity + time windows).

    objective:
      - "min_distance": minimize total distance + tardiness penalty
      - "min_vehicles_then_distance": lexicographic (vehicles first, then distance)
    """
    rng = random.Random(seed)
    n = D.shape[0] - 1

    BIG_M = 1e6  # for lexicographic scalarization

    def decode(order: List[int]):
        routes, cost = split_tour_into_routes_timewindow(
            order, demands, capacity, time_windows, service_times, D
        )
        vehicles_used = len(routes)
        return routes, vehicles_used, cost

    def decode_cost(order: List[int]) -> float:
        _, used, dist_pen = decode(order)
        if objective == "min_distance":
            return dist_pen
        else:  # vehicles first, then distance
            return used * BIG_M + dist_pen

    # population init on permutations (1..n)
    def init_pop():
        base = list(range(1, n + 1))
        pop = []
        for _ in range(n_pop):
            rng.shuffle(base)
            pop.append(base[:])
        return pop

    def ordered_crossover(p1, p2):
        a, b = sorted(rng.sample(range(n), 2))
        hole = set(p1[a:b + 1])
        child = [None] * n
        child[a:b + 1] = p1[a:b + 1]
        fill = [x for x in p2 if x not in hole]
        j = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[j]
                j += 1
        return child

    def mutate_swap(ch):
        i, j = rng.sample(range(n), 2)
        ch[i], ch[j] = ch[j], ch[i]

    # GA loop
    pop = init_pop()
    fitness = [1.0 / (decode_cost(ind) + 1e-9) for ind in pop]
    best = pop[int(np.argmax(fitness))][:]
    best_cost = 1.0 / max(fitness)
    history = [(0, best_cost)]
    t0 = time.time()

    for it in range(1, iters + 1):
        new_pop = [best[:]]  # elitism
        while len(new_pop) < n_pop:
            a, b = rng.sample(range(n_pop), 2)
            c, d = rng.sample(range(n_pop), 2)
            p1 = pop[a] if fitness[a] > fitness[b] else pop[b]
            p2 = pop[c] if fitness[c] > fitness[d] else pop[d]
            child = ordered_crossover(p1, p2) if rng.random() < cx_rate else p1[:]
            if rng.random() < mut_rate:
                mutate_swap(child)
            new_pop.append(child)

        pop = new_pop
        fitness = [1.0 / (decode_cost(ind) + 1e-9) for ind in pop]
        cur_idx = int(np.argmax(fitness))
        cur_cost = 1.0 / fitness[cur_idx]
        if cur_cost < best_cost:
            best_cost = cur_cost
            best = pop[cur_idx][:]
        history.append((it, best_cost))

    runtime = time.time() - t0

    # Decode best chromosome to explicit routes and stats
    routes, vehicles_used, dist_pen = decode(best)
    out_best_cost = dist_pen if objective == "min_distance" else vehicles_used * BIG_M + dist_pen

    return {
        "routes": routes,
        "vehicles_used": vehicles_used,
        "best_cost": out_best_cost,
        "distance_penalized": dist_pen,
        "history": history,
        "runtime": runtime,
        "chromosome": best,
        "objective": objective,
    }
