
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils import (
    generate_simulated_coords, pairwise_distance_matrix,
)
from tsp_sa import run_sa
from tsp_aco import run_aco
from tsp_tabu import run_tabu
from vrp_ga import run_vrp_ga

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

STYLE_CYCLE = [
    dict(color="#e41a1c", weight=4, dash_array=""),        # red solid
    dict(color="#377eb8", weight=4, dash_array="8,6"),     # blue dashed
    dict(color="#4daf4a", weight=4, dash_array="2,8"),     # green dotted
    dict(color="#984ea3", weight=4, dash_array="12,6,2,6"),# purple dash-dot
    dict(color="#ff7f00", weight=4, dash_array="4,4"),     # orange dashed
    dict(color="#a65628", weight=4, dash_array="1,6"),     # brown sparse dots
]

# ==== Pretty VRP helpers ====
def vrp_route_distance(route, D):
    """Distance of a single vehicle route that starts/ends at depot 0."""
    dist = 0.0
    prev = 0
    for c in route:
        dist += D[prev, c]
        prev = c
    dist += D[prev, 0]
    return dist

def vrp_routes_pretty_table(routes, D, city_names=None, demands=None):
    """
    Build a DataFrame: one row per vehicle.
    - routes: List[List[int]] customers only (no depot)
    - D: full (n+1)x(n+1) distance matrix (0 is depot)
    - city_names: optional list of node names, index 0 is depot name ('Depot'), 1..n are customers
    - demands: optional list of demands same length as city_names (demands[0] must be 0)
    """
    rows = []
    for vid, r in enumerate(routes):
        # path text with names
        path_nodes = [0] + r + [0]
        if city_names:
            def name(i): return city_names[i]
            path_str = " \u2192 ".join(name(i) for i in path_nodes)  # → arrows
        else:
            path_str = " \u2192 ".join(str(i) for i in path_nodes)

        # load & stops
        load = sum((demands[c] if demands else 0) for c in r)
        stops = len(r)

        # distance
        dist = vrp_route_distance(r, D)

        rows.append({
            "Vehicle": f"Vehicle {vid+1}",
            "Path": path_str,
            "Stops": stops,
            "Load": load,
            "Distance": round(dist, 3),
        })
    import pandas as pd
    return pd.DataFrame(rows)

def explain_vrp_params():
    import streamlit as st
    with st.expander("What do these parameters mean?"):
        st.markdown(
            "- **Vehicle capacity**: maximum total demand a single vehicle can carry on its route.\n"
            "- **Demand min / max**: when generating customers, each customer's demand is a random integer in this range.\n"
            "- **GA iterations**: number of generations the Genetic Algorithm will evolve; higher = slower but potentially better.\n"
            "- **Random seed**: fixes the random choices (cities sampling / demands / GA) for reproducible results."
        )

def style_for(idx: int) -> dict:
    return STYLE_CYCLE[idx % len(STYLE_CYCLE)]

st.set_page_config(page_title="TSP/VRP Optimizer", layout="wide")
st.title("TSP & VRP Optimizer Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Simulated City TSP", "Real City TSP", "Simulated City VRP", "Real City VRP"])

def plot_tour_overlay(coords, routes_dict, title="TSP Tours (overlay)"):
    xs, ys = coords[:,0], coords[:,1]
    plt.figure()
    plt.scatter(xs, ys, s=20, label="cities")
    for name, route in routes_dict.items():
        r = route + [route[0]]
        plt.plot(xs[r], ys[r], label=name, linewidth=1.6)
    plt.title(title); plt.legend(); plt.grid(True)
    st.pyplot(plt.gcf())

def plot_convergence(history, title="Convergence curve"):
    iters = [h[0] for h in history]; bests = [h[1] for h in history]
    plt.figure(); plt.plot(iters, bests)
    plt.xlabel("Iteration"); plt.ylabel("Best cost so far"); plt.title(title)
    st.pyplot(plt.gcf())

def make_folium_map(coords_latlon, routes_dict=None, is_vrp=False, vrp_routes=None):
    if not HAS_FOLIUM:
        st.warning("Folium is not available. Install `folium` and `streamlit_folium`."); return
    lats = coords_latlon[:,0]; lons = coords_latlon[:,1]
    center = [float(np.mean(lats)), float(np.mean(lons))]
    if 17 <= center[0] <= 54 and 73 <= center[1] <= 135:
        m = folium.Map(location=[35.0, 103.8], zoom_start=4)
    else:
        m = folium.Map(location=center, zoom_start=4)
    for i, (lat, lon) in enumerate(coords_latlon):
        folium.CircleMarker(location=[lat, lon], radius=3, tooltip=str(i)).add_to(m)
    if is_vrp and vrp_routes is not None:
        for idx, route in enumerate(vrp_routes):
            pts = [[coords_latlon[0,0], coords_latlon[0,1]]]
            for c in route: pts.append([coords_latlon[c,0], coords_latlon[c,1]])
            pts.append([coords_latlon[0,0], coords_latlon[0,1]])
            folium.PolyLine(locations=pts, weight=3, tooltip=f"Route {idx+1}").add_to(m)
    elif routes_dict:
        for name, route in routes_dict.items():
            pts = [[coords_latlon[i,0], coords_latlon[i,1]] for i in (route + [route[0]])]
            folium.PolyLine(locations=pts, weight=3, tooltip=name).add_to(m)
    st_folium(m, height=520, width=None)

# ---------- Simulated TSP ----------
with tab1:
    st.header("Simulated City TSP")
    n = st.slider("Number of cities n (<= 800)", 10, 800, 100, step=10)
    metric = st.selectbox("Distance metric", ["euclidean"])
    seed = st.number_input("Random seed", value=0, step=1)
    st.markdown("**All three algorithms (SA/ACO/Tabu) will be executed.** The overlay figure shows their tours on the same plot; other comparison plots remain.")
    if st.button("Run TSP (Simulated)"):
        coords = generate_simulated_coords(n, seed=seed)
        D = pairwise_distance_matrix(coords, metric=metric)
        sa = run_sa(D, seed=seed); aco = run_aco(D, seed=seed); tabu = run_tabu(D, seed=seed)
        st.subheader("Summary")
        df = pd.DataFrame([
            {"Algorithm":"Simulated Annealing","Best cost":sa["best_cost"],"Runtime (s)":sa["runtime"],"Converged after iter ≥":sa["time_convergence_iter"]},
            {"Algorithm":"Ant Colony","Best cost":aco["best_cost"],"Runtime (s)":aco["runtime"],"Converged after iter ≥":aco["time_convergence_iter"]},
            {"Algorithm":"Tabu Search","Best cost":tabu["best_cost"],"Runtime (s)":tabu["runtime"],"Converged after iter ≥":tabu["time_convergence_iter"]},
        ]); st.dataframe(df, use_container_width=True)
        st.subheader("Overlay of routes (SA vs ACO vs Tabu)")
        plot_tour_overlay(coords, {"SA":sa["route"],"ACO":aco["route"],"Tabu":tabu["route"]}, title="Simulated TSP: Overlay of Tours")
        st.subheader("Convergence curves")
        c1,c2,c3 = st.columns(3)
        with c1: plot_convergence(sa["history"], "SA Convergence")
        with c2: plot_convergence(aco["history"], "ACO Convergence")
        with c3: plot_convergence(tabu["history"], "Tabu Convergence")

# Load built-in China cities
china_df = pd.read_csv("data/china_cities.csv")

# keep state across reruns
if "real_tsp_state" not in st.session_state:
    st.session_state.real_tsp_state = None
if "real_tsp_selected" not in st.session_state:
    st.session_state.real_tsp_selected = ["Beijing","Shanghai","Guangzhou","Shenzhen"]
if "real_tsp_n" not in st.session_state:
    st.session_state.real_tsp_n = 0
if "real_tsp_seed" not in st.session_state:
    st.session_state.real_tsp_seed = 0

# ---------- Real TSP with built-in city list ----------
with tab2:
    st.header("Real City TSP (Built-in China Cities)")
    st.markdown(
        "Select cities from the built-in list or specify a number to sample randomly. "
        "Distances use great-circle (km). Routes are drawn on a China map. "
        "**Minimum 4 cities.**"
    )
    all_names = list(china_df["name"].values)

    selected = st.multiselect(
        "Select cities (≥ 4)",
        all_names,
        default=st.session_state.real_tsp_selected,
    )
    n_random = st.number_input(
        "Or sample N cities randomly (≥ 4)",
        min_value=0, max_value=len(all_names),
        value=st.session_state.real_tsp_n, step=1,
    )
    seed2 = st.number_input(
        "Random seed",
        value=st.session_state.real_tsp_seed, step=1, key="seed2",
    )

    if st.button("Run TSP (Real from built-in)", key="run_real_tsp"):
        rng = np.random.default_rng(int(seed2))
        if n_random and n_random >= 4:
            chosen = list(rng.choice(all_names, size=int(n_random), replace=False))
        else:
            chosen = selected

        # remember UI state
        st.session_state.real_tsp_selected = selected
        st.session_state.real_tsp_n = int(n_random)
        st.session_state.real_tsp_seed = int(seed2)

        if len(chosen) < 4:
            st.session_state.real_tsp_state = {"error": "Please choose at least 4 cities."}
        else:
            sub = china_df.set_index("name").loc[chosen].reset_index()
            coords_latlon = sub[["lat","lon"]].to_numpy()
            D = pairwise_distance_matrix(coords_latlon, metric="haversine")

            sa = run_sa(D)
            aco = run_aco(D)
            tabu = run_tabu(D)

            st.session_state.real_tsp_state = {
                "sub": sub.to_dict(orient="list"),
                "coords": coords_latlon.tolist(),  # store as list in session
                "sa": sa, "aco": aco, "tabu": tabu,
            }

    # ---- render from state (persists after rerun) ----
    state = st.session_state.real_tsp_state
    if state:
        if "error" in state:
            st.error(state["error"])
        else:
            sub = pd.DataFrame(state["sub"])
            coords = np.array(state["coords"])
            sa, aco, tabu = state["sa"], state["aco"], state["tabu"]

            st.subheader("Summary")
            dfres = pd.DataFrame([
                {"Algorithm":"Simulated Annealing","Best cost (km)":sa["best_cost"],"Runtime (s)":sa["runtime"],"Converged after iter ≥":sa["time_convergence_iter"]},
                {"Algorithm":"Ant Colony","Best cost (km)":aco["best_cost"],"Runtime (s)":aco["runtime"],"Converged after iter ≥":aco["time_convergence_iter"]},
                {"Algorithm":"Tabu Search","Best cost (km)":tabu["best_cost"],"Runtime (s)":tabu["runtime"],"Converged after iter ≥":tabu["time_convergence_iter"]},
            ])
            st.dataframe(dfres, use_container_width=True)

            st.subheader("Overlay of routes on China map")
            try:
                import folium
                from streamlit_folium import st_folium

                m = folium.Map(location=[35.0, 103.8], zoom_start=4)

                for i, (lat, lon) in enumerate(coords):
                    folium.CircleMarker(
                        [lat, lon], radius=4,
                        color="#555555", fill=True, fill_opacity=0.9,
                        tooltip=f"{i}: {sub['name'].iloc[i]}"
                    ).add_to(m)

                start_idx = sa["route"][0]
                folium.Marker(
                    [coords[start_idx,0], coords[start_idx,1]],
                    tooltip=f"Start: {sub['name'].iloc[start_idx]}",
                    icon=folium.Icon(color="green", icon="play", prefix="fa")
                ).add_to(m)

                algo_routes = [("SA", sa["route"]), ("ACO", aco["route"]), ("Tabu", tabu["route"])]
                for idx, (name, route) in enumerate(algo_routes):
                    fg = folium.FeatureGroup(name=f"{name} route", show=True)
                    pts = [[coords[i,0], coords[i,1]] for i in (route + [route[0]])]
                    folium.PolyLine(
                        locations=pts,
                        tooltip=name,
                        **style_for(idx)
                    ).add_to(fg)
                    fg.add_to(m)

                folium.LayerControl(collapsed=False).add_to(m)
                st_folium(m, height=560, width=None)

            except Exception:
                st.info("Map view unavailable. Please install `folium` and `streamlit-folium`.")


            st.subheader("Convergence curves")
            c1,c2,c3 = st.columns(3)
            with c1:
                it,best = zip(*sa["history"])
                plt.figure(); plt.plot(it,best); plt.title("SA Convergence"); plt.xlabel("Iteration"); plt.ylabel("Best cost"); st.pyplot(plt.gcf())
            with c2:
                it,best = zip(*aco["history"])
                plt.figure(); plt.plot(it,best); plt.title("ACO Convergence"); plt.xlabel("Iteration"); plt.ylabel("Best cost"); st.pyplot(plt.gcf())
            with c3:
                it,best = zip(*tabu["history"])
                plt.figure(); plt.plot(it,best); plt.title("Tabu Convergence"); plt.xlabel("Iteration"); plt.ylabel("Best cost"); st.pyplot(plt.gcf())


# ---------- Simulated VRP ----------
# ---------- Simulated VRP (overlay plot like your TSP figure) ----------
with tab3:
    st.header("Simulated City VRP (GA)")

    # ---- Controls ----
    n3 = st.slider("Number of customers (excluding depot)", 5, 200, 40, step=5)
    capacity = st.number_input("Vehicle capacity", value=50, step=1)
    demand_min = st.number_input("Demand min (per customer)", value=1, step=1)
    demand_max = st.number_input("Demand max (per customer)", value=10, step=1)
    iters_ga = st.number_input("GA iterations", value=800, step=50)
    seed3 = st.number_input("Random seed", value=0, step=1, key="seed3_sim_vrp")

    # ---- Parameter help ----
    with st.expander("What do these parameters mean?"):
        st.markdown(
            "- **Number of customers**: customers to serve (node 0 is the depot, not counted here).\n"
            "- **Vehicle capacity**: maximum total demand a single vehicle can carry.\n"
            "- **Demand min / max**: each customer's demand is a random integer in this range.\n"
            "- **GA iterations**: generations for the Genetic Algorithm (more = slower but potentially better).\n"
            "- **Random seed**: fixes random choices for reproducibility."
        )

    # ---- Pretty helpers (local to this block) ----
    def _route_distance(route, D):
        """Distance of one vehicle route (0 -> ... -> 0)."""
        import numpy as _np
        d = 0.0
        prev = 0
        for c in route:
            d += D[prev, c]
            prev = c
        d += D[prev, 0]
        return d

    def _routes_table(routes, D, demands):
        """Return a human-readable DataFrame: one row per vehicle."""
        import pandas as _pd
        rows = []
        for vid, r in enumerate(routes):
            path_str = " \u2192 ".join(["Depot"] + [f"C{c}" for c in r] + ["Depot"])  # Depot → Cx → ... → Depot
            rows.append({
                "Vehicle": f"Vehicle {vid+1}",
                "Path": path_str,
                "Stops": len(r),
                "Load": int(sum(demands[c] for c in r)),
                "Distance": round(_route_distance(r, D), 3),
            })
        return _pd.DataFrame(rows)

    def _plot_vrp_overlay(coords, routes):
        """
        Matplotlib overlay:
        - depot (index 0) as a star
        - customers as dots
        - one polyline per vehicle: 0 -> route -> 0
        """
        import matplotlib.pyplot as _plt
        import numpy as _np

        xs, ys = coords[:, 0], coords[:, 1]

        _plt.figure()
        # customers
        _plt.scatter(xs[1:], ys[1:], s=30, label="customers")
        # depot
        _plt.scatter([xs[0]], [ys[0]], marker="*", s=180, label="depot")

        # color/linestyle cycles for visual distinction
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        linestyles = ["-", "--", ":", "-.", (0, (8, 6)), (0, (2, 8)), (0, (12, 6, 2, 6)), (0, (4, 4))]

        for i, r in enumerate(routes):
            path = [0] + r + [0]
            _plt.plot(xs[path], ys[path],
                      linestyle=linestyles[i % len(linestyles)],
                      linewidth=2,
                      label=f"Vehicle {i+1}",
                      color=colors[i % len(colors)])

        _plt.title("Simulated VRP: Overlay of Routes")
        _plt.xlabel("x"); _plt.ylabel("y")
        _plt.legend()
        _plt.grid(True)
        _plt.axis("equal")  # keep aspect ratio similar
        st.pyplot(_plt.gcf())

    # ---- Run button ----
    if st.button("Run VRP (Simulated)"):
        # coordinates (include depot at index 0)
        coords = generate_simulated_coords(n3 + 1, seed=seed3)
        D = pairwise_distance_matrix(coords, metric="euclidean")

        # demands
        if demand_max < demand_min:
            demand_max = demand_min
        rng = np.random.default_rng(int(seed3))
        demands = [0] + list(rng.integers(int(demand_min), int(demand_max) + 1, size=n3))

        # GA solve
        res = run_vrp_ga(D, demands, int(capacity), iters=int(iters_ga), seed=int(seed3))

        # Summary line
        st.success(f"Best total distance: {res['best_cost']:.3f} | Runtime: {res['runtime']:.2f}s")

        # Human-readable table
        table = _routes_table(res["routes"], D, demands)
        st.dataframe(table, use_container_width=True)

        # Overlay plot (like your TSP figure)
        _plot_vrp_overlay(coords, res["routes"])

# ---------- Real VRP using built-in cities ----------
# ---------- Real VRP using built-in China cities ----------
with tab4:
    st.header("Real City VRP (GA, Built-in China Cities)")

    # ---- parameter help ----
    with st.expander("What do these parameters mean?"):
        st.markdown(
            "- **First city is the depot** (demand = 0). Others are customers.\n"
            "- **Vehicle capacity**: max total demand a single vehicle can carry.\n"
            "- **Demand min / max**: each customer's demand is a random integer in this range.\n"
            "- **GA iterations**: generations for the Genetic Algorithm; higher = slower but potentially better.\n"
            "- **Random seed**: fixes city sampling, demands and GA randomness for reproducible runs."
        )

    # ---- keep state across reruns ----
    if "real_vrp_state" not in st.session_state:
        st.session_state.real_vrp_state = None

    # ---- style helpers for distinct polylines (both Matplotlib & Folium) ----
    STYLE_CYCLE = [
        dict(color="#e41a1c", weight=4, dash_array=""),         # red solid
        dict(color="#377eb8", weight=4, dash_array="8,6"),      # blue dashed
        dict(color="#4daf4a", weight=4, dash_array="2,8"),      # green dotted
        dict(color="#984ea3", weight=4, dash_array="12,6,2,6"), # purple dash-dot
        dict(color="#ff7f00", weight=4, dash_array="4,4"),      # orange dashed
        dict(color="#a65628", weight=4, dash_array="1,6"),      # brown sparse dots
        dict(color="#f781bf", weight=4, dash_array="3,3,10,3"), # pink mixed
        dict(color="#999999", weight=4, dash_array=""),         # gray solid
    ]
    def style_for(i: int) -> dict:
        return STYLE_CYCLE[i % len(STYLE_CYCLE)]

    # ---- pretty outputs ----
    def _route_distance(route, D):
        d = 0.0
        prev = 0
        for c in route:
            d += D[prev, c]
            prev = c
        d += D[prev, 0]
        return d

    def _routes_table(routes, D, city_names, demands):
        rows = []
        for vid, r in enumerate(routes):
            path_str = " \u2192 ".join([city_names[0]] + [city_names[c] for c in r] + [city_names[0]])
            rows.append({
                "Vehicle": f"Vehicle {vid+1}",
                "Path": path_str,
                "Stops": len(r),
                "Load": int(sum(demands[c] for c in r)),
                "Distance (km)": round(_route_distance(r, D), 3),
            })
        return pd.DataFrame(rows)

    def _plot_vrp_overlay(coords, routes):
        """Matplotlib overlay like the TSP figure: depot star, customers dots, one colored/linestyled polyline per vehicle."""
        xs, ys = coords[:, 0], coords[:, 1]
        plt.figure()
        # customers + depot
        plt.scatter(xs[1:], ys[1:], s=30, label="customers")
        plt.scatter([xs[0]], [ys[0]], marker="*", s=180, label="depot")
        # colors/linestyles
        color_cycle = ["C0","C1","C2","C3","C4","C5","C6","C7"]
        linestyle_cycle = ["-","--",":","-.", (0,(8,6)), (0,(2,8)), (0,(12,6,2,6)), (0,(4,4))]
        for i, r in enumerate(routes):
            path = [0] + r + [0]
            plt.plot(xs[path], ys[path],
                     color=color_cycle[i % len(color_cycle)],
                     linestyle=linestyle_cycle[i % len(linestyle_cycle)],
                     linewidth=2,
                     label=f"Vehicle {i+1}")
        plt.title("Real VRP: Overlay of Routes (projected coords)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(); plt.grid(True); plt.axis("equal")
        st.pyplot(plt.gcf())

    # ---- UI: pick cities ----
    all_names = list(china_df["name"].values)
    DEPOT_DEFAULT = "Beijing"  

    default_names = [DEPOT_DEFAULT, "Shanghai", "Guangzhou", "Shenzhen"]
    selected = st.multiselect(
        "Select cities (≥ 2, the first is depot)",
        all_names,
        default=default_names,
        key="vrp_names"
    )

    n_random = st.number_input(
        "Or sample N cities randomly (≥ 2)",
        min_value=0, max_value=len(all_names), value=15, step=1, key="vrp_n"
    )

    seed4      = st.number_input("Random seed", value=0, step=1, key="seed4")
    capacity4  = st.number_input("Vehicle capacity", value=25, step=1, key="cap4")   # 从 25 起步，更容易触发多车
    demand_min = st.number_input("Demand min (customers)", value=6, step=1, key="dmin")
    demand_max = st.number_input("Demand max (customers)", value=12, step=1, key="dmax")
    iters_ga   = st.number_input("GA iterations", value=800, step=50, key="ga_iters")

    if st.button("Run VRP (Real from built-in)", key="run_real_vrp"):
        rng = np.random.default_rng(int(seed4))

        if n_random and n_random >= 2:
            # fix depot as Beijing if possible
            candidates = [n for n in all_names if n != DEPOT_DEFAULT]
            k = int(n_random) - 1
            if k > len(candidates):
                k = len(candidates)
            chosen_customers = list(rng.choice(candidates, size=k, replace=False))
            chosen = [DEPOT_DEFAULT] + chosen_customers
        else:
            # use selected list as is, but ensure depot is first if present
            chosen = selected[:]
            if len(chosen) >= 1 and chosen[0] != DEPOT_DEFAULT and DEPOT_DEFAULT in chosen:
                chosen.remove(DEPOT_DEFAULT)
                chosen = [DEPOT_DEFAULT] + chosen

        if len(chosen) < 2:
            st.session_state.real_vrp_state = {"error": "Please choose at least 2 cities."}
        else:
            sub = china_df.set_index("name").loc[chosen].reset_index()
            coords_latlon = sub[["lat","lon"]].to_numpy()
            D = pairwise_distance_matrix(coords_latlon, metric="haversine")

            n_customers = len(chosen) - 1
            if demand_max < demand_min:
                demand_max = demand_min
            cust_demands = list(rng.integers(int(demand_min), int(demand_max)+1, size=n_customers))
            demands = [0] + cust_demands

            res = run_vrp_ga(D, demands, int(capacity4), iters=int(iters_ga), seed=int(seed4))

            st.session_state.real_vrp_state = {
                "city_names": list(sub["name"].values),     # index 0 is depot
                "coords": coords_latlon.tolist(),
                "demands": demands,
                "result": res,
            }


    # ---- render from state (persists after reruns) ----
    state = st.session_state.real_vrp_state
    if state:
        if "error" in state:
            st.error(state["error"])
        else:
            city_names   = state["city_names"]
            coords_latln = np.array(state["coords"])
            demands      = state["demands"]
            res          = state["result"]
            # Recompute D from stored coords (haversine km)
            D = pairwise_distance_matrix(coords_latln, metric="haversine")

            st.success(f"Best total distance: {res['best_cost']:.3f} | Runtime: {res['runtime']:.2f}s")

            # human-friendly table
            table = _routes_table(res["routes"], D, city_names=city_names, demands=demands)
            st.dataframe(table, use_container_width=True)

            # Folium map with distinct styles + layer control
            try:
                import folium
                from streamlit_folium import st_folium

                m = folium.Map(location=[35.0, 103.8], zoom_start=4)

                # depot marker
                folium.Marker(
                    [coords_latln[0,0], coords_latln[0,1]],
                    tooltip=f"Depot: {city_names[0]}",
                    icon=folium.Icon(color="red", icon="home", prefix="fa")
                ).add_to(m)

                # customers
                for i in range(1, len(coords_latln)):
                    folium.CircleMarker(
                        [coords_latln[i,0], coords_latln[i,1]], radius=4,
                        color="#555555", fill=True, fill_opacity=0.9,
                        tooltip=f"{i}: {city_names[i]} | demand={demands[i]}"
                    ).add_to(m)

                # one layer per vehicle with different color/linestyle
                for idx, route in enumerate(res["routes"]):
                    fg = folium.FeatureGroup(name=f"Vehicle {idx+1}", show=True)
                    pts = [[coords_latln[0,0], coords_latln[0,1]]]
                    for c in route:
                        pts.append([coords_latln[c,0], coords_latln[c,1]])
                    pts.append([coords_latln[0,0], coords_latln[0,1]])
                    folium.PolyLine(
                        locations=pts, tooltip=f"Vehicle {idx+1}", **style_for(idx)
                    ).add_to(fg)
                    fg.add_to(m)

                folium.LayerControl(collapsed=False).add_to(m)
                st_folium(m, height=560, width=None)
            except Exception:
                st.info("Map view is unavailable because `folium` and/or `streamlit-folium` is not installed.")
