
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import solver & parser from your VRPTW module
from vrp_ga_timewindow import (
    load_vrptw_from_txt,
    run_vrp_ga_timewindow,
)

# ------------------------------- Visualization --------------------------------

def plot_vrptw_routes(coords: np.ndarray,
                      routes: List[List[int]],
                      title: str = "VRPTW: Overlay of Routes"):
    """
    Spatial overlay:
      - depot (index 0) as a star
      - customers as dots
      - one polyline per vehicle: 0 -> route -> 0
    """
    xs, ys = coords[:, 0], coords[:, 1]
    plt.figure()
    # customers & depot
    plt.scatter(xs[1:], ys[1:], s=30, label="customers")
    plt.scatter([xs[0]], [ys[0]], marker="*", s=180, label="depot")

    colors = ["C0","C1","C2","C3","C4","C5","C6","C7"]
    linestyles = ["-","--",":","-.", (0,(8,6)), (0,(2,8)), (0,(12,6,2,6)), (0,(4,4))]
    for i, r in enumerate(routes):
        path = [0] + r + [0]
        plt.plot(xs[path], ys[path],
                 color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)],
                 linewidth=2,
                 label=f"Vehicle {i+1}")
    plt.title(title)
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True); plt.axis("equal")
    return plt.gcf()


def _schedule_for_route(route: List[int],
                        D: np.ndarray,
                        time_windows: List[Tuple[float, float]],
                        service_times: List[float]):
    """
    Compute arrival, start, finish for each stop in a single route.
    Start at depot t=0, travel time = D distance (speed=1).
    Returns list of dicts per stop and return time to depot.
    """
    sched = []
    t = 0.0
    prev = 0
    for c in route:
        travel = D[prev, c]
        arrival = t + travel
        ready, due = time_windows[c]
        start = max(arrival, ready)   # wait if early
        finish = start + service_times[c]
        sched.append({
            "cust": c,
            "arrival": arrival,
            "start": start,
            "finish": finish,
            "ready": ready,
            "due": due,
            "wait": max(0.0, start - arrival),
            "tardiness": max(0.0, start - due),
        })
        t = finish
        prev = c
    back = t + D[prev, 0]
    return sched, back


def plot_vrptw_timeline(routes: List[List[int]],
                        D: np.ndarray,
                        time_windows: List[Tuple[float, float]],
                        service_times: List[float],
                        title: str = "VRPTW: Timeline (Time Windows, Waiting, Service)"):
    """
    Per-vehicle timeline:
      - One row per vehicle.
      - For each stop: draw time window [ready,due] as a light band,
        and the actual service interval [start,finish] as a solid bar.
      - Waiting is the gap between arrival and start (not filled).
      - Tardiness appears as service bar extending beyond the window band.
    """
    nV = len(routes)
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.9*nV)))
    y_ticks = []
    y_labels = []
    colors = ["C0","C1","C2","C3","C4","C5","C6","C7"]

    y = 0
    for vid, r in enumerate(routes):
        sched, back_time = _schedule_for_route(r, D, time_windows, service_times)
        # draw each stop
        for st in sched:
            ready, due = st["ready"], st["due"]
            start, finish = st["start"], st["finish"]
            cust = st["cust"]
            # time window band
            ax.fill_between([ready, due], y-0.35, y+0.35,
                            color="lightgray", alpha=0.5, linewidth=0)
            # service bar
            ax.barh(y, finish - start, left=start, height=0.5,
                    color=colors[vid % len(colors)], edgecolor="black", alpha=0.9)
            # label
            ax.text(finish + 0.5, y, f"C{cust}", va="center", fontsize=8)
        # return-time marker (optional)
        ax.plot([back_time, back_time], [y-0.45, y+0.45], color="k", linestyle=":", linewidth=1)

        y_ticks.append(y)
        y_labels.append(f"Vehicle {vid+1}")
        y += 1

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("time")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":")
    fig.tight_layout()
    return fig

# ------------------------------- Runner (test) --------------------------------

def main():
    parser = argparse.ArgumentParser(description="VRPTW visualizer + runner (GA + Time Windows)")
    parser.add_argument("--instance", type=str, required=True,
                        help="Path to Solomon-style VRPTW .txt file (e.g., C101.txt)")
    parser.add_argument("--objective", type=str, default="min_vehicles_then_distance",
                        choices=["min_vehicles_then_distance", "min_distance"],
                        help="Optimization objective")
    parser.add_argument("--iters", type=int, default=1200, help="GA iterations")
    parser.add_argument("--n_pop", type=int, default=200, help="GA population size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save figures")

    args = parser.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load instance
    coords, D, demands, tw, svc, cap, k_suggest = load_vrptw_from_txt(args.instance)
    print(f"Loaded: {args.instance} | customers={len(coords)-1} | cap={cap} | suggested_vehicles={k_suggest}")

    # 2) Solve by GA (time windows)
    res = run_vrp_ga_timewindow(
        D, demands, cap, tw, svc,
        n_pop=args.n_pop, iters=args.iters,
        objective=args.objective, seed=args.seed
    )
    routes = res["routes"]
    print(f"Vehicles used: {res['vehicles_used']} | Best cost: {res['best_cost']:.3f} | Runtime: {res['runtime']:.2f}s")

    # 3) Spatial overlay
    fig1 = plot_vrptw_routes(coords, routes, title=f"VRPTW Overlay ({Path(args.instance).name})")
    f1 = outdir / f"{Path(args.instance).stem}_overlay.png"
    fig1.savefig(f1, dpi=150)
    print(f"Saved: {f1}")

    # 4) Timeline (time windows)
    fig2 = plot_vrptw_timeline(routes, D, tw, svc,
                               title=f"VRPTW Timeline ({Path(args.instance).name})")
    f2 = outdir / f"{Path(args.instance).stem}_timeline.png"
    fig2.savefig(f2, dpi=150)
    print(f"Saved: {f2}")

if __name__ == "__main__":
    main()
