
from typing import Dict, Any, List, Tuple
import numpy as np
import random, math, time
from utils import route_length, greedy_init_tsp, two_opt, ConvergenceTracker

def run_sa(D: np.ndarray,
           iters: int = 20000,
           T0: float = 100.0,
           alpha: float = 0.9995,
           seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    n = D.shape[0]
    route = greedy_init_tsp(D, start=0)
    best_route = route[:]
    best_cost = route_length(route, D)
    cur_cost = best_cost
    T = T0

    tracker = ConvergenceTracker()
    tracker.update(0, best_cost)

    t0 = time.time()
    for it in range(1, iters+1):
        if n >= 4:
            pool = range(1, n-1)
        else:
            pool = range(0, n)

        i, k = sorted(rng.sample(pool, 2))
        new_route = two_opt(route, i, k)
        new_cost = route_length(new_route, D)
        delta = new_cost - cur_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            route = new_route
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_route = route[:]
        tracker.update(it, best_cost)
        T *= alpha
    runtime = time.time() - t0
    return {"route": best_route, "best_cost": best_cost, "history": tracker.history,
            "time_convergence_iter": tracker.time_convergence_iter, "runtime": runtime}
