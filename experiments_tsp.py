
"""
Run experiments comparing SA, ACO, and Tabu over varying instance sizes (up to 800 cities).
Generates instances, runs algorithms, records best costs, runtimes, and time-convergence iters.
Saves summary CSV and plots into outputs/.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from utils import generate_simulated_coords, pairwise_distance_matrix
from tsp_sa import run_sa
from tsp_aco import run_aco
from tsp_tabu import run_tabu

# sizes=(50, 100, 200, 400, 800)
def run_suite(sizes=(50, 100, 200), seed=0, outdir=Path("outputs")):
    rows = []
    for n in sizes:
        seed = int(time.time() * 1000) % (2**32 - 1)
        random.seed(seed)
        coords = generate_simulated_coords(n, seed=seed)
        D = pairwise_distance_matrix(coords, metric="euclidean")

        print(f"=== N={n} ===")
        # SA
        sa = run_sa(D, iters=max(20000, n*200), T0=100, alpha=0.9995, seed=seed)
        # ACO
        aco = run_aco(D, n_ants=min(40, max(10, n//20)), iters=max(200, n//2), seed=seed)
        # Tabu
        tabu = run_tabu(D, iters=max(2000, n*5), tabu_tenure=50, seed=seed)

        for name, res in [("SA", sa), ("ACO", aco), ("Tabu", tabu)]:
            rows.append({
                "n": n,
                "algo": name,
                "best_cost": res["best_cost"],
                "runtime_sec": res["runtime"],
                "time_convergence_iter": res["time_convergence_iter"],
            })

    df = pd.DataFrame(rows)
    outdir.mkdir(exist_ok=True, parents=True)
    csv_path = outdir / "tsp_compare_summary.csv"
    df.to_csv(csv_path, index=False)

    # Plot runtime vs n
    plt.figure()
    for algo in df["algo"].unique():
        sub = df[df["algo"] == algo]
        plt.plot(sub["n"], sub["runtime_sec"], marker="o", label=algo)
    plt.xlabel("Cities (n)"); plt.ylabel("Runtime (s)"); plt.title("TSP runtime vs n")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / "tsp_runtime_vs_n.png")

    # Plot runtime vs n
    plt.figure()
    for algo in df["algo"].unique():
        sub = df[df["algo"] == algo]
        plt.plot(sub["n"], sub["runtime_sec"], marker="o", label=algo)
    plt.xlabel("Cities (n)"); plt.ylabel("Runtime (s)"); plt.title("TSP runtime vs n")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / "tsp_runtime_vs_n.png")

    # Plot convergence iter vs n
    plt.figure()
    for algo in df["algo"].unique():
        sub = df[df["algo"] == algo]
        plt.plot(sub["n"], sub["time_convergence_iter"], marker="o", label=algo)
    plt.xlabel("Cities (n)"); plt.ylabel("Convergence iteration")
    plt.title("TSP time-convergence iteration vs n")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / "tsp_convergence_iter_vs_n.png")

    return csv_path

if __name__ == "__main__":
    run_suite()
