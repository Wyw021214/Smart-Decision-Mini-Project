import numpy as np
import random, time
from typing import List, Dict, Any
from utils import split_tour_into_routes, vrp_routes_cost

def run_vrp_ga(D: np.ndarray,
               demands: List[int],
               capacity: int,
               n_pop: int = 200,
               iters: int = 1000,
               cx_rate: float = 0.8,
               mut_rate: float = 0.2,
               seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    n = D.shape[0] - 1

    def decode_cost(order: List[int]) -> float:
        routes = split_tour_into_routes(order, demands, capacity)
        return vrp_routes_cost(routes, D)

    def init_pop():
        pop = []
        base = list(range(1, n+1))
        for _ in range(n_pop):
            rng.shuffle(base); pop.append(base[:])
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
    routes = split_tour_into_routes(best, demands, capacity)
    return {"routes": routes, "best_cost": best_cost, "history": history, "runtime": runtime}
