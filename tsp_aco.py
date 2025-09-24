
import numpy as np
import time, random, math
from typing import Dict, Any
from utils import route_length, ConvergenceTracker

def run_aco(D: np.ndarray,
            n_ants: int = 20,
            iters: int = 500,
            alpha: float = 1.0,
            beta: float = 5.0,
            rho: float = 0.5,
            q: float = 100.0,
            seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    n = D.shape[0]
    eta = 1.0 / (D + 1e-12)
    np.fill_diagonal(eta, 0.0)
    tau = np.ones((n, n))
    np.fill_diagonal(tau, 0.0)

    tracker = ConvergenceTracker()
    best_cost = float("inf")
    best_route = None

    t0 = time.time()
    for it in range(iters):
        all_routes = []
        all_costs = []
        for ant in range(n_ants):
            start = rng.randrange(n)
            visited = [start]
            unvisited = set(range(n)); unvisited.remove(start)
            while unvisited:
                i = visited[-1]
                probs = []
                denom = 0.0
                for j in unvisited:
                    val = (tau[i, j] ** alpha) * (eta[i, j] ** beta)
                    probs.append((j, val)); denom += val
                r = rng.random() * denom
                cum = 0.0; choice = None
                for j, val in probs:
                    cum += val
                    if r <= cum:
                        choice = j; break
                if choice is None: choice = next(iter(unvisited))
                visited.append(choice); unvisited.remove(choice)
            cost = route_length(visited, D)
            all_routes.append(visited); all_costs.append(cost)

        tau *= (1 - rho)
        idx = int(np.argmin(all_costs))
        route = all_routes[idx]; cost = all_costs[idx]
        for a, b in zip(route, route[1:]+[route[0]]):
            tau[a, b] += q / cost; tau[b, a] += q / cost
        if cost < best_cost:
            best_cost = cost; best_route = route
        tracker.update(it, best_cost)

    runtime = time.time() - t0
    return {"route": best_route, "best_cost": best_cost, "history": tracker.history,
            "time_convergence_iter": tracker.time_convergence_iter, "runtime": runtime}
