import os
import json
import yaml
import numpy as np
import rioxarray as rxr

def da_to_numpy(da):
    arr = da.squeeze(drop=True).to_numpy().astype(float)
    nod = da.rio.nodata
    if nod is not None:
        arr[arr == nod] = np.nan
    return arr

def main(cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    p = cfg["paths"]
    os.makedirs(p["out_dir"], exist_ok=True)

    # Read and align rasters
    ref = rxr.open_rasterio(p["index_tif"])
    ref = ref.squeeze(drop=True)  # [y,x]
    px_m = abs(ref.rio.transform()[0])

    dist_da = rxr.open_rasterio(p["distance_tif"]).rio.reproject_match(ref)
    clas_da = rxr.open_rasterio(p["class_tif"]).rio.reproject_match(ref)

    index = da_to_numpy(ref)
    dist  = da_to_numpy(dist_da)
    clas  = da_to_numpy(clas_da)

    # Valid mask
    elig = np.zeros_like(clas, dtype=bool)
    for c in cfg["eligible_classes"]:
        elig |= (clas == c)
    mask_valid = np.isfinite(index) & np.isfinite(dist) & np.isfinite(clas)
    valid = mask_valid & elig

    # Access bonus
    dmin, dmax = np.nanmin(dist), np.nanmax(dist)
    dist_norm = (dist - dmin) / (dmax - dmin + 1e-9)
    access_bonus = np.exp(-cfg["access_decay"] * dist_norm)

    # Type weights
    type_weight = np.zeros_like(clas, dtype=float)
    for k, v in cfg["type_weight"].items():
        type_weight[clas == int(k)] = float(v)

    # Potential (0..1)
    potential = 1.0 - index  # higher potential -> more uplift
    potential = np.clip(potential, 0.0, 1.0)

    V_ha_carbon = (
        cfg["delta_tCO2e_per_ha"]
        * cfg["years"]
        * cfg["price_per_tCO2e"]
    )

    # Optional co-benefits $/ha by class
    coben_ha = np.zeros_like(clas, dtype=float)
    for k, v in cfg.get("cobenefit_ha", {}).items():
        coben_ha[clas == int(k)] = float(v)

    # Per-pixel factors
    pixel_ha = (px_m * px_m) / 10000.0
    V_pix_carbon = V_ha_carbon * pixel_ha
    V_pix_coben  = coben_ha * pixel_ha

    # Benefit in USD/pixel
    benefit = potential * type_weight * access_bonus * (V_pix_carbon + V_pix_coben)

    # Cost in USD/pixel
    cost_ha = np.zeros_like(clas, dtype=float)
    for k, v in cfg["cost_ha"].items():
        cost_ha[clas == int(k)] = float(v)

    alpha = cfg["alpha_per_km"]
    dist_km = dist / 1000.0
    cost = cost_ha * pixel_ha * (1.0 + alpha * dist_km)
    cost = np.where(np.isnan(cost), 0.0, cost)

    # Compute BLM default
    mean_cost_valid = float(np.nanmean(cost[valid])) if np.any(valid) else 0.0
    blm = mean_cost_valid * float(cfg["blm_multiplier_mean_cost"])

    # Save arrays compactly
    np.savez_compressed(
        os.path.join(p["out_dir"], "inputs.npz"),
        benefit=benefit,
        cost=cost,
        valid=valid.astype(np.uint8),  # save as 0/1 to reduce size
        px_m=np.array([px_m], dtype=float),
        neighbors=np.array([cfg["neighbors"]], dtype=int),
    )

    # Save a small profile for writing GeoTIFF later
    profile = {
        "crs": str(ref.rio.crs),
        "transform": list(ref.rio.transform()),
        "width": int(ref.rio.width),
        "height": int(ref.rio.height),
        "dtype": "uint8",
        "nodata": 0
    }
    with open(os.path.join(p["out_dir"], "profile.json"), "w") as f:
        json.dump(profile, f)

    scalars = {
        "budget_usd": cfg["budget_usd"],
        "min_spend_usd": cfg["min_spend_usd"],
        "blm_default": blm
    }
    with open(os.path.join(p["out_dir"], "scalars.json"), "w") as f:
        json.dump(scalars, f, indent=2)

    print("Prepared arrays saved to:", p["out_dir"])
    print(f"px_m={px_m:.3f} m, pixel_ha={pixel_ha:.5f} ha")
    print(f"Valid eligible pixels: {int(valid.sum())} (~{valid.sum()*pixel_ha:.2f} ha)")
    print(f"BLM default (USD/m): {blm:.4f}")

if __name__ == "__main__":
    main()
