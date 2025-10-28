import os
import json
import yaml
import numpy as np
import rasterio
from rasterio.transform import Affine
import pyomo.environ as pyo
import matplotlib.pyplot as plt

def build_edges(idx_map, H, W, px_m, neighbors=4):
    E = []
    w_edge = []
    if neighbors == 4:
        nbrs = [(1,0),(0,1),(-1,0),(0,-1)]
    elif neighbors == 8:
        nbrs = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    else:
        raise ValueError("neighbors must be 4 or 8")

    for r in range(H):
        for c in range(W):
            u = idx_map[r, c]
            if u < 0:
                continue
            for dr, dc in nbrs:
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W:
                    v = idx_map[rr, cc]
                    if v >= 0 and v > u:
                        E.append((u, v))
                        if neighbors == 4:
                            w = px_m if (dr==0 or dc==0) else px_m
                        else:
                            w = (px_m*np.sqrt(2.0)) if (dr!=0 and dc!=0) else px_m
                        w_edge.append(w)
    return E, np.array(w_edge, dtype=float)

def write_geotiff(path, array_u8, profile_json):
    # array_u8: 2D uint8 (0/1)
    profile = dict(profile_json)
    transform = Affine(*profile["transform"])
    meta = {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "count": 1,
        "dtype": "uint8",
        "crs": profile["crs"],
        "transform": transform,
        "nodata": profile.get("nodata", 0),
        "compress": "LZW"
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array_u8, 1)

def main(cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    out_dir = p["out_dir"]

    data = np.load(os.path.join(out_dir, "inputs.npz"))
    benefit = data["benefit"]
    cost    = data["cost"]
    valid   = data["valid"].astype(bool)
    px_m    = float(data["px_m"][0])
    neighbors = int(data["neighbors"][0])

    with open(os.path.join(out_dir, "scalars.json"), "r") as f:
        scalars = json.load(f)
    with open(os.path.join(out_dir, "profile.json"), "r") as f:
        profile = json.load(f)

    H, W = benefit.shape
    # Map valid pixels to ids
    idx = -np.ones((H, W), dtype=int)
    flat_valid = np.flatnonzero(valid)
    idx.flat[flat_valid] = np.arange(len(flat_valid))
    P = list(range(len(flat_valid)))

    b_vec = benefit.flat[flat_valid].astype(float)
    c_vec = cost.flat[flat_valid].astype(float)

    # Edges
    E, w_edge = build_edges(idx, H, W, px_m, neighbors=neighbors)

    # Pyomo model
    m = pyo.ConcreteModel()
    m.P = pyo.Set(initialize=P)
    m.E = pyo.Set(initialize=list(range(len(E))))
    m.benefit = pyo.Param(m.P, initialize={i: float(b_vec[i]) for i in P}, mutable=False)
    m.cost    = pyo.Param(m.P, initialize={i: float(c_vec[i]) for i in P}, mutable=False)
    m.u       = pyo.Param(m.E, initialize={e: E[e][0] for e in m.E}, mutable=False)
    m.v       = pyo.Param(m.E, initialize={e: E[e][1] for e in m.E}, mutable=False)
    m.w       = pyo.Param(m.E, initialize={e: float(w_edge[e]) for e in m.E}, mutable=False)

    m.x = pyo.Var(m.P, within=pyo.Binary)
    m.z = pyo.Var(m.E, bounds=(0,1))

    m.B   = pyo.Param(initialize=float(cfg["budget_usd"]), mutable=False)
    m.minB = pyo.Param(initialize=float(cfg["min_spend_usd"]), mutable=False)
    m.blm = pyo.Param(initialize=float(scalars["blm_default"]), mutable=True)
    CARD_MIN = 5000
    CARD_MAX = 10000
    m.CardMin = pyo.Constraint(expr=sum(m.x[i] for i in m.P) >= CARD_MIN)
    m.CardMax = pyo.Constraint(expr=sum(m.x[i] for i in m.P) <= CARD_MAX)

    def obj_rule(m):
        return sum(m.benefit[i]*m.x[i] for i in m.P) - m.blm*sum(m.w[e]*m.z[e] for e in m.E)
    m.Obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    def budget_upper(m):
        return sum(m.cost[i]*m.x[i] for i in m.P) <= m.B
    m.BudgetU = pyo.Constraint(rule=budget_upper)

    def budget_lower(m):
        return sum(m.cost[i]*m.x[i] for i in m.P) >= m.minB
    m.BudgetL = pyo.Constraint(rule=budget_lower)

    # Linearize z_e = |x_u - x_v|
    def z_abs1(m, e): 
        u, v = m.u[e], m.v[e]
        return m.z[e] >= m.x[u] - m.x[v]
    def z_abs2(m, e): 
        u, v = m.u[e], m.v[e]
        return m.z[e] >= m.x[v] - m.x[u]
    def z_abs3(m, e): 
        u, v = m.u[e], m.v[e]
        return m.z[e] <= m.x[u] + m.x[v]
    def z_abs4(m, e): 
        u, v = m.u[e], m.v[e]
        return m.z[e] <= 2 - (m.x[u] + m.x[v])

    m.Z1 = pyo.Constraint(m.E, rule=z_abs1)
    m.Z2 = pyo.Constraint(m.E, rule=z_abs2)
    m.Z3 = pyo.Constraint(m.E, rule=z_abs3)
    m.Z4 = pyo.Constraint(m.E, rule=z_abs4)

    # Solve
    solver = pyo.SolverFactory("highs")

    # ---- HPC solver configuration ----
    solver.options["threads"] = 16           # use half your 32 cores
    solver.options["time_limit"] = 1800      # stop after 30 min
    solver.options["mip_rel_gap"] = 0.01     # 1% tolerance for faster convergence
    solver.options["presolve"] = "on"
    solver.options["log_file"] = os.path.join(out_dir, "highs_solver.log")

    print("Running HiGHS solver (16 threads, 30-min limit)...")
    res = solver.solve(m, tee=True)

    # Log summary
    with open(os.path.join(out_dir, "highs_summary.txt"), "w") as f:
        f.write(str(res))


    # Solution
    x_sol = np.zeros(len(P), dtype=np.uint8)
    for i in P:
        x_sol[i] = int(pyo.value(m.x[i]) > 0.5)

    selected = np.zeros_like(valid, dtype=np.uint8)
    selected.flat[flat_valid] = x_sol

    # Diagnostics
    total_benefit = float(np.sum(b_vec[x_sol==1]))
    total_cost    = float(np.sum(c_vec[x_sol==1]))
    # boundary length term value (meters weighted by blm)
    z_vals = np.array([pyo.value(m.z[e]) for e in m.E], dtype=float)
    boundary_len = float(np.sum(w_edge * z_vals))

    pixel_ha = (px_m*px_m)/10000.0
    print(f"Selected pixels: {x_sol.sum()}  |  Area (ha): {x_sol.sum()*pixel_ha:.2f}")
    print(f"Total cost: ${total_cost:,.0f}  |  Total benefit: ${total_benefit:,.0f}")
    print(f"Boundary length (m): {boundary_len:,.1f}")
    print(f"Objective value (USD): {pyo.value(m.Obj):,.0f}")

    # Write selection GeoTIFF (0/1 = treated/not treated grassland)
    out_tif = os.path.join(out_dir, "selection_01.tif")
    write_geotiff(out_tif, selected, profile)
    print("Wrote:", out_tif)

    # Quick plot
    plt.figure(figsize=(8,6))
    plt.imshow(selected, interpolation="nearest")
    plt.title("Selected pixels (1=treated grassland)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "selection_01.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
