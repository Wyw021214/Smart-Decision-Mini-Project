
import math
import time
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def pairwise_distance_matrix(coords: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=float)
    if metric == "euclidean":
        for i in range(n):
            for j in range(i+1, n):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                D[i, j] = D[j, i] = d
    elif metric == "haversine":
        for i in range(n):
            for j in range(i+1, n):
                d = haversine_km(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
                D[i, j] = D[j, i] = d
    else:
        raise ValueError("Unknown metric")
    return D

def route_length(route: List[int], D: np.ndarray) -> float:
    n = len(route)
    total = 0.0
    for i in range(n):
        total += D[route[i], route[(i+1) % n]]
    return total

def two_opt(route: List[int], i: int, k: int) -> List[int]:
    return route[:i] + list(reversed(route[i:k+1])) + route[k+1:]

def greedy_init_tsp(D: np.ndarray, start: int = 0) -> List[int]:
    n = D.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    route = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        unvisited.remove(nxt)
        route.append(nxt)
        cur = nxt
    return route

class ConvergenceTracker:
    def __init__(self):
        self.best_cost = float("inf")
        self.best_iter = -1
        self.history = []

    def update(self, iter_idx: int, cost: float):
        if cost < self.best_cost - 1e-12:
            self.best_cost = cost
            self.best_iter = iter_idx
        self.history.append((iter_idx, self.best_cost))

    @property
    def time_convergence_iter(self) -> int:
        return self.best_iter + 1

def load_real_cities(csv_path: str, max_n: Optional[int] = None):
    df = pd.read_csv(csv_path)
    required = {"name", "lat", "lon"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    if max_n is not None:
        df = df.iloc[:max_n].copy()
    coords = df[["lat","lon"]].to_numpy()
    return coords, df

def generate_simulated_coords(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n,2)) * 100.0

def split_tour_into_routes(order: List[int], demands: List[int], capacity: int):
    routes = []
    cur = []
    load = 0
    for c in order:
        d = demands[c]
        if load + d <= capacity:
            cur.append(c); load += d
        else:
            if cur: routes.append(cur)
            cur = [c]; load = d
    if cur: routes.append(cur)
    return routes

def vrp_routes_cost(routes: List[List[int]], D: np.ndarray) -> float:
    total = 0.0
    for r in routes:
        prev = 0
        for c in r:
            total += D[prev, c]
            prev = c
        total += D[prev, 0]
    return total


