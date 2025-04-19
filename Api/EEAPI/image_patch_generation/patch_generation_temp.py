import re
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.windows import Window
from shapely.geometry import box, Point


def plot_sar_with_spill_and_patches(
    image_path,
    shapefile_path,
    spills_df,
    patch_size=512,
    nrows=2,
    ncols=2,
    land_threshold=0.35,
    show_plot=True
):
    """
    1) Reads a Sentinel-1 image and a shapefile, plus a spill dataset.
    2) Finds the spill point from the image name (via 'Original index').
    3) Plots the base map with the spill point and region boundary (optional).
    4) Creates patches in a simple grid around the spill point:
       - (2*nrows + 1) rows, centered on spill point row
       - For each row's 'center patch', if it intersects the region > land_threshold,
         keep shifting left by patch_size until it's under threshold or we run out of attempts.
       - Then for ncols times, keep shifting left for new patches.
         Each time, skip patches with region overlap > threshold.

    Parameters
    ----------
    image_path : str
        Full path to the Sentinel-1 GeoTIFF image.
    shapefile_path : str
        Full path to the DEPARTAMENTOS.shp file.
    spills_df : pd.DataFrame
        Must include ['LAT', 'LON', 'REGION', 'Original index'] columns.
    patch_size : int
        Patch dimension in pixels (512 by default).
    nrows : int
        Number of rows above/below the spill point's row.
    ncols : int
        Number of patches to the LEFT of the center patch in each row.
    land_threshold : float
        If (land overlap area / total patch area) > land_threshold, shift left or skip.
    show_plot : bool
        If True, shows the map and patch outlines in a matplotlib figure.

    Returns
    -------
    dict
        A dictionary with keys:
          - 'image_index'
          - 'transform'
          - 'crs'
          - 'bounds'
          - 'region_polygon'
          - 'spill_point'
          - 'patch_polygons': List of Shapely boxes (map coords) for the final patches
          - 'patch_windows':  Corresponding rasterio Windows for reading those patches
    """
    # -------------- 1) Parse the image index --------------
    image_filename = image_path.split("/")[-1].split("\\")[-1]
    match = re.match(r"(\d+)_", image_filename)
    if not match:
        raise ValueError("Could not parse index from filename.")
    image_index = int(match.group(1))
    print(f"[INFO] Image index: {image_index}")

    # -------------- 2) Convert DataFrame -> GeoDataFrame --------------
    if not isinstance(spills_df, gpd.GeoDataFrame):
        spills_df["geometry"] = spills_df.apply(
            lambda row: Point(row["LAT"], row["LON"]), axis=1
        )
        spills_gdf = gpd.GeoDataFrame(spills_df, geometry="geometry", crs="EPSG:4326")
    else:
        spills_gdf = spills_df

    # -------------- 3) Filter for the relevant spill --------------
    filtered_spills = spills_gdf[spills_gdf["Original index"] == image_index]
    if filtered_spills.empty:
        raise ValueError(f"No spills found for index {image_index}")
    print(f"[INFO] Found {len(filtered_spills)} spill(s) for index {image_index}")

    # We'll just pick the first if multiple
    spill_row = filtered_spills.iloc[0]

    # -------------- 4) Open the raster, read transform & bounds --------------
    with rasterio.open(image_path) as src:
        img = src.read(1)  # Not strictly necessary if we only want patches
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        width, height = src.width, src.height

    # -------------- 5) Load and reproject shapefile --------------
    target_region = spill_row["REGION"]
    peru_gdf = gpd.read_file(shapefile_path)
    # Filter for region
    peru_gdf = peru_gdf[peru_gdf["NOMBDEP"] == target_region]
    if peru_gdf.empty:
        raise ValueError(f"Region '{target_region}' not found in shapefile.")
    if peru_gdf.crs is None:
        peru_gdf.set_crs(epsg=4326, inplace=True)

    # Reproject everything
    if peru_gdf.crs != crs:
        peru_gdf = peru_gdf.to_crs(crs)
        filtered_spills = filtered_spills.to_crs(crs)

    region_polygon = peru_gdf.geometry.iloc[0]
    if region_polygon is None or region_polygon.is_empty:
        raise ValueError("The region polygon is empty or invalid.")

    # -------------- 6) Reprojected spill point --------------
    spill_point = filtered_spills.geometry.iloc[0]

    # -------------- 7) Convert patch_size from pixel distances to map units --------------
    # We'll assume uniform resolution from transform (approx).
    res_x = transform.a  # pixel width (map units / pixel)
    res_y = transform.e  # often negative
    mean_res = (abs(res_x) + abs(res_y)) / 2.0
    patch_width_map = patch_size * mean_res  # in map units

    # -------------- 8) Build a function to get patch polygon from a center (map coords) --------------
    from shapely.geometry import box as shapely_box
    def make_patch_polygon(cx, cy):
        # half-size in map units
        half = patch_width_map / 2.0
        return shapely_box(cx - half, cy - half, cx + half, cy + half)

    # -------------- 9) Intersection ratio check --------------
    def land_overlap_ratio(patch_poly, region_poly):
        intersection_area = patch_poly.intersection(region_poly).area
        patch_area = patch_poly.area
        if patch_area == 0:
            return 0
        return intersection_area / patch_area

    # -------------- 10) Convert spill point to pixel coords --------------
    with rasterio.open(image_path) as src:
        spill_px, spill_py = ~src.transform * (spill_point.x, spill_point.y)
        # So (spill_px, spill_py) is the pixel coordinate of the spill

    # -------------- 11) For each row, find an acceptable center patch --------------
    patch_windows = []
    patch_polygons = []

    # We'll store them in a top-down approach, so row offsets go from +nrows down to -nrows
    row_offsets = range(nrows, -nrows - 1, -1)  # e.g. if nrows=2 => [2,1,0,-1,-2]

    with rasterio.open(image_path) as src:
        for row_off in row_offsets:
            # row_off indicates how many *patch_height* steps we move up/down
            # Negative means down, so let's define new center row in pixel space:
            this_row_py = spill_py + (row_off * patch_size)

            # Step (A): find a good center patch in the same column as the spill
            # We'll test shifting left multiple times if overlap > threshold.
            # But user wants the first patch to also shift left if needed. We'll define
            # a small loop to shift left if needed, up to ncols attempts.

            attempts_left = ncols  # how many shifts left we can do
            found_center = False
            center_px = spill_px
            center_py = this_row_py

            # We'll do a while loop: check the overlap ratio, if > land_threshold, shift left 1 patch
            while attempts_left >= 0:
                # Construct patch polygon in map coords
                center_map = src.transform * (center_px, center_py)
                patch_poly = make_patch_polygon(*center_map)
                ratio = land_overlap_ratio(patch_poly, region_polygon)
                if ratio <= land_threshold:
                    found_center = True
                    break
                else:
                    # shift left
                    center_px -= patch_size
                    attempts_left -= 1

            if not found_center:
                # If we still can't find a patch center with < threshold, we skip this row
                print(f'[WARN] Row offset {row_off}: Could not find center patch under threshold.')
                continue

            # We found a center patch => record it
            center_map_x, center_map_y = src.transform * (center_px, center_py)
            center_poly = make_patch_polygon(center_map_x, center_map_y)

            # Build the rasterio window
            center_row_off = int(center_py - patch_size / 2)
            center_col_off = int(center_px - patch_size / 2)
            # Check bounds
            if (center_row_off < 0 or center_col_off < 0 or
                (center_row_off + patch_size) > src.height or
                (center_col_off + patch_size) > src.width):
                print(f'[WARN] Center patch out of raster bounds; skipping row {row_off}')
                continue

            patch_windows.append(Window(
                col_off=center_col_off, row_off=center_row_off,
                width=patch_size, height=patch_size
            ))
            patch_polygons.append(center_poly)

            # Step (B): for ncols times, create additional patches to the left
            prev_px = center_px
            for c in range(ncols):
                new_px = prev_px - patch_size  # shift left
                new_map_x, new_map_y = src.transform * (new_px, center_py)
                new_poly = make_patch_polygon(new_map_x, new_map_y)

                ratio = land_overlap_ratio(new_poly, region_polygon)
                if ratio > land_threshold:
                    print(f'[INFO] Skipping patch col={c+1} in row offset {row_off} due to ratio={ratio:.2f}')
                    # Optionally keep shifting left if you want (but user asked for 1 shot)
                    continue

                # Build a Window for the new patch
                new_row_off = int(center_py - patch_size / 2)
                new_col_off = int(new_px - patch_size / 2)
                if (new_row_off < 0 or new_col_off < 0 or
                    (new_row_off + patch_size) > src.height or
                    (new_col_off + patch_size) > src.width):
                    print(f'[WARN] Patch col={c+1} out of bounds; skipping.')
                    continue

                patch_windows.append(Window(
                    col_off=new_col_off, row_off=new_row_off,
                    width=patch_size, height=patch_size
                ))
                patch_polygons.append(new_poly)

                # update for next iteration
                prev_px = new_px

    # -------------- 12) Optionally plot the result --------------
    if show_plot:
        # Re-open the raster just for plotting
        with rasterio.open(image_path) as src:
            fig, ax = plt.subplots(figsize=(10, 10))
            show((src, 1), ax=ax, cmap='gray')

            # Plot region boundary
            gpd.GeoDataFrame({'geometry':[region_polygon]}, crs=peru_gdf.crs)\
               .boundary.plot(ax=ax, edgecolor='red', linewidth=1.5)

            # Plot spill point
            gpd.GeoDataFrame({'geometry':[spill_point]}, crs=peru_gdf.crs)\
               .plot(ax=ax, color='cyan', markersize=30, zorder=5)

            # Plot each patch polygon
            for patch_poly in patch_polygons:
                x_patch, y_patch = patch_poly.exterior.xy
                ax.plot(x_patch, y_patch, color='yellow', linewidth=1.5)
            ax.set_xlim(bounds.left, bounds.right)
            ax.set_ylim(bounds.bottom, bounds.top)
            ax.set_title(f"Patches for Spill (Index={image_index})")
            plt.tight_layout()
            plt.show()

import pandas as pd

# 1) Load your spill Excel

df = pd.read_excel(r'D:\AndreJuarez\Code_OilSpill\Apis\EEAPI\Codigo_final_preproc_data\LastImages_full_oilspills_66.xlsx')

# Paths
img_path = r"G:\Mi unidad\GEE_img_66_512_final\7_1_2015-03-16_S1_GRD_VV.tif"
shp_path = r'D:/AndreJuarez/Code_OilSpill/Apis/last_dance/DEPARTAMENTOS.shp'

result = plot_sar_with_spill_and_patches(
    image_path=img_path,
    shapefile_path=shp_path,
    spills_df=df,
    patch_size=512,
    nrows=2,          # 2 rows above + 2 below + center = 5 total
    ncols=2,          # up to 2 patches to the left
    land_threshold=0.35,
    show_plot=True    # or False if you only want metadata
)
