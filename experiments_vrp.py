"""
experiments_VRP.py

Run experiments for VRP GA over:
  - varying number of customers,
  - homogeneous vs heterogeneous fleets,
  - objectives: min_distance vs min_vehicles_then_distance.

Writes:
  - outputs/vrp_experiments_summary.csv
  - outputs/vrp_runtime_vs_n.png
  - outputs/vrp_vehicles_vs_n.png
  - outputs/vrp_distance_vs_n.png
"""

from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import generate_simulated_coords, pairwise_distance_matrix
from vrp_ga import run_vrp_ga  # must be the updated version supporting hetero + objectives


# --------- instance generation ---------
def generate_random_vrp(n_customers: int,
                        seed: int = 0,
                        demand_low: int = 1,
                        demand_high: int = 10):
    """
    Build a synthetic VRP instance:
      - n_customers + 1 depot (index 0)
      - Euclidean distances
      - integer demands in [demand_low, demand_high]
    """
    coords = generate_simulated_coords(n_customers + 1, seed=seed)
    D = pairwise_distance_matrix(coords, metric="euclidean")
    rng = np.random.default_rng(seed)
    demands = [0] + rng.integers(demand_low, demand_high + 1, size=n_customers).tolist()
    return coords, D, demands


# --------- main experiment suite ---------
def run_suite(
    sizes=(20, 50, 100, 150),
    objectives=("min_distance", "min_vehicles_then_distance"),
    # Fleet configurations to compare:
    # - Homogeneous: {"label": "...", "vehicle_capacities": 30}
    # - Heterogeneous: {"label": "...", "vehicle_capacities": [50,40,40,20]}
    fleet_configs=(
        {"label": "Homo_C30", "vehicle_capacities": 30},                 # homogeneous cap=30
        {"label": "Hetero_50-40-40-20", "vehicle_capacities": [50,40,40,20]},  # heterogeneous
    ),
    demand_low=6,
    demand_high=12,
    iters_scale=20,     # GA iterations ~= iters_scale * n_customers (min 500)
    n_pop=200,
    seed=0,
    outdir=Path("outputs")
):
    rows = []
    outdir.mkdir(parents=True, exist_ok=True)

    for n in sizes:
        # new seed per size to avoid correlation
        seed_n = int(time.time() * 1000) % (2**32 - 1)
        random.seed(seed_n)

        coords, D, demands = generate_random_vrp(n, seed=seed_n,
                                                 demand_low=demand_low,
                                                 demand_high=demand_high)

        ga_iters = max(500, int(iters_scale * n))  # scale iterations with problem size

        print(f"\n=== VRP n_customers={n} ===")
        for cfg in fleet_configs:
            label = cfg["label"]
            caps = cfg["vehicle_capacities"]

            for obj in objectives:
                print(f"  - Fleet={label:>18s} | Obj={obj:>26s} | iters={ga_iters}")

                res = run_vrp_ga(
                    D, demands,
                    vehicle_capacities=caps,
                    objective=obj,
                    # You can pass max_vehicles (homogeneous only) if desired:
                    # max_vehicles=None,
                    n_pop=n_pop,
                    iters=ga_iters,
                    seed=seed_n
                )

                # Distance (without the lexicographic BIG-M). For fair plotting,
                # re-compute total distance from routes:
                # (If heterogeneous and overflow penalty exists, this is pure distance.)
                dist_only = _routes_total_distance(res["routes"], D)

                rows.append({
                    "n_customers": n,
                    "fleet_label": label,
                    "objective": obj,
                    "best_cost": res["best_cost"],
                    "runtime_sec": res["runtime"],
                    "vehicles_used": res.get("vehicles_used", len(res["routes"])),
                    "distance_only": dist_only,
                })

    df = pd.DataFrame(rows)
    csv_path = outdir / "vrp_experiments_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")

    # --- PLOTS ---
    # Runtime vs n
    plt.figure()
    for (label, obj), sub in df.groupby(["fleet_label", "objective"]):
        sub = sub.sort_values("n_customers")
        plt.plot(sub["n_customers"], sub["runtime_sec"], marker="o", label=f"{label} | {obj}")
    plt.xlabel("Customers (n)")
    plt.ylabel("Runtime (s)")
    plt.title("VRP: runtime vs number of customers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "vrp_runtime_vs_n.png")

    # Vehicles used vs n
    plt.figure()
    for (label, obj), sub in df.groupby(["fleet_label", "objective"]):
        sub = sub.sort_values("n_customers")
        plt.plot(sub["n_customers"], sub["vehicles_used"], marker="o", label=f"{label} | {obj}")
    plt.xlabel("Customers (n)")
    plt.ylabel("Vehicles used")
    plt.title("VRP: vehicles used vs number of customers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "vrp_vehicles_vs_n.png")

    # Distance vs n (pure total distance)
    plt.figure()
    for (label, obj), sub in df.groupby(["fleet_label", "objective"]):
        sub = sub.sort_values("n_customers")
        plt.plot(sub["n_customers"], sub["distance_only"], marker="o", label=f"{label} | {obj}")
    plt.xlabel("Customers (n)")
    plt.ylabel("Total distance")
    plt.title("VRP: total distance vs number of customers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "vrp_distance_vs_n.png")

    return csv_path


# --- helper to recompute pure distance from routes (for plotting) ---
def _routes_total_distance(routes, D):
    total = 0.0
    for r in routes:
        prev = 0
        for c in r:
            total += D[prev, c]
            prev = c
        total += D[prev, 0]
    return float(total)


if __name__ == "__main__":
    run_suite()
