
import numpy as np
import time, random
from typing import Dict, Any
from utils import route_length, two_opt, greedy_init_tsp, ConvergenceTracker

def run_tabu(D: np.ndarray,
             iters: int = 2000,
             tabu_tenure: int = 50,
             seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    n = D.shape[0]
    cur = greedy_init_tsp(D, start=0)
    cur_cost = route_length(cur, D)
    best = cur[:]; best_cost = cur_cost
    tabu = {}
    tracker = ConvergenceTracker(); tracker.update(0, best_cost)

    t0 = time.time()
    for it in range(1, iters+1):
        best_move = None; best_move_cost = float("inf")
        for _ in range(200):
            i, k = sorted(rng.sample(range(1, n-1), 2))
            move = (i, k)
            cand = two_opt(cur, i, k)
            cost = route_length(cand, D)
            if move in tabu and cost >= best_cost: continue
            if cost < best_move_cost:
                best_move_cost = cost; best_move = (move, cand)
        if best_move is None:
            i, k = sorted(rng.sample(range(1, n-1), 2))
            cur = two_opt(cur, i, k); cur_cost = route_length(cur, D)
        else:
            (i, k), cand = best_move
            cur = cand; cur_cost = best_move_cost
            tabu[(i, k)] = it + tabu_tenure
            for m in list(tabu.keys()):
                if tabu[m] <= it: del tabu[m]
            if cur_cost < best_cost:
                best_cost = cur_cost; best = cur[:]
        tracker.update(it, best_cost)

    runtime = time.time() - t0
    return {"route": best, "best_cost": best_cost, "history": tracker.history,
            "time_convergence_iter": tracker.time_convergence_iter, "runtime": runtime}
