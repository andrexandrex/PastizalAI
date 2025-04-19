import re
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import box, Point
from rasterio.plot import show
from rasterio.windows import Window
from tiff_vis import plot_sar_with_spill
import pandas as pd


def plot_sar_with_spill_and_patches(
    image_path, 
    shapefile_path, 
    spills_df, 
    patch_size=512, 
    nrows=2,          # number of rows above and below the oil spill row
    ncols=2,          # number of additional adjacent columns to the left (per row)
    land_threshold=0.35, 
    show_plot=True
):
    """
    Build a grid of patches starting from the oil spill:
      - For each row (above and below the spill), find the first patch by moving
        a little to the left until the patch has ≤ land_threshold overlap.
      - Once the first valid patch for a row is found, add additional adjacent patches
        to the left (each exactly patch_size apart) provided they also meet the threshold.
    """
    # Get base data from your existing plotting function
    sar_data = plot_sar_with_spill(image_path, shapefile_path, spills_df, show_plot=False)

    transform = sar_data['transform']
    crs = sar_data['crs']
    region_polygon = sar_data['region_gdf'].geometry.iloc[0]
    spill_point = sar_data['spill_gdf'].geometry.iloc[0]
    image_index = sar_data['image_index']
    bounds = sar_data['bounds']
    # Determine map resolution and patch polygon dimensions in map units.
    res_x = transform.a
    res_y = transform.e
    mean_res = (abs(res_x) + abs(res_y)) / 2.0
    patch_width_map = patch_size * mean_res

    def make_patch_polygon(cx, cy):
        half = patch_width_map / 2.0
        return box(cx - half, cy - half, cx + half, cy + half)

    def land_overlap_ratio(patch_poly, region_poly):
        intersection_area = patch_poly.intersection(region_poly).area
        patch_area = patch_poly.area
        return intersection_area / patch_area if patch_area > 0 else 0

    patch_windows = []
    patch_polygons = []

    # Open the image to work with pixel coordinates
    with rasterio.open(image_path) as src:
        # Convert the spill coordinates (map units) to pixel coordinates.
        spill_px, spill_py = ~src.transform * (spill_point.x, spill_point.y)

        # Define row offsets relative to the oil spill row.
        # For example, if nrows=2 then we process rows at: +2, +1, 0, -1, -2.
        row_offsets = range(nrows, -nrows - 1, -1)

        # Define a small step (in pixel units) to nudge left until we get a valid patch.
        small_step = patch_size * 0.1  # adjust this factor as needed

        for row_off in row_offsets:
            # Determine the center Y pixel coordinate for this row.
            center_py = spill_py + (row_off * patch_size)

            # For each row, start from the oil spill's x coordinate.
            candidate_px = spill_px

            # Try nudging left in small steps until the patch meets the land criteria.
            found_first = False
            max_attempts = 50  # Avoid infinite loops; adjust as needed.
            attempts = 0

            while attempts < max_attempts:
                candidate_map = src.transform * (candidate_px, center_py)
                candidate_poly = make_patch_polygon(*candidate_map)
                ratio = land_overlap_ratio(candidate_poly, region_polygon)
                if ratio <= land_threshold:
                    found_first = True
                    break
                candidate_px -= small_step
                attempts += 1

            # If no valid patch was found in this row, skip it.
            if not found_first:
                continue

            # Check that the patch is fully within the image bounds.
            center_row_off = int(center_py - patch_size / 2)
            center_col_off = int(candidate_px - patch_size / 2)
            if (
                center_row_off < 0 or center_col_off < 0 or
                (center_row_off + patch_size) > src.height or
                (center_col_off + patch_size) > src.width
            ):
                continue

            # Add the first valid patch for this row.
            patch_windows.append(Window(center_col_off, center_row_off, patch_size, patch_size))
            patch_polygons.append(candidate_poly)

            # Now build the grid to the left of this first patch.
            prev_px = candidate_px
            for col in range(1, ncols + 1):
                new_px = prev_px - patch_size  # adjacent patch: full patch width to the left
                new_map = src.transform * (new_px, center_py)
                new_poly = make_patch_polygon(*new_map)
                ratio = land_overlap_ratio(new_poly, region_polygon)
                if ratio > land_threshold:
                    # If the adjacent patch exceeds land threshold, do not add it.
                    # (Alternatively, you could attempt a small nudge here as well, but
                    #  the requirement is to only add strictly adjacent patches.)
                    break

                new_row_off = int(center_py - patch_size / 2)
                new_col_off = int(new_px - patch_size / 2)
                if (
                    new_row_off < 0 or new_col_off < 0 or
                    (new_row_off + patch_size) > src.height or
                    (new_col_off + patch_size) > src.width
                ):
                    break

                patch_windows.append(Window(new_col_off, new_row_off, patch_size, patch_size))
                patch_polygons.append(new_poly)
                prev_px = new_px

    # Optionally show the results.
    if show_plot:
        with rasterio.open(image_path) as src:
            fig, ax = plt.subplots(figsize=(10, 10))
            show((src, 1), ax=ax, cmap='gray')
            # Plot the coast boundary (region) and the spill location.
            sar_data['region_gdf'].boundary.plot(ax=ax, edgecolor='red', linewidth=1.5)
            sar_data['spill_gdf'].plot(ax=ax, color='cyan', markersize=30, zorder=5)
            # Plot each patch.
            for patch_poly in patch_polygons:
                x_patch, y_patch = patch_poly.exterior.xy
                ax.plot(x_patch, y_patch, color='yellow', linewidth=1.5)
            ax.set_title(f"Patches for Spill (Index={image_index})")
            ax.set_xlim(bounds.left, bounds.right)
            ax.set_ylim(bounds.bottom, bounds.top)
            plt.tight_layout()
            plt.show()

    return {
        'image_index': image_index,
        'transform': transform,
        'crs': crs,
        'region_polygon': region_polygon,
        'spill_point': spill_point,
        'patch_polygons': patch_polygons,
        'patch_windows': patch_windows
    }



df = pd.read_excel(r'D:\AndreJuarez\Code_OilSpill\Apis\EEAPI\Codigo_final_preproc_data\LastImages_full_oilspills_66.xlsx')

# Paths
img_path = r"G:\Mi unidad\GEE_img_66_512_final\7_1_2015-03-16_S1_GRD_VV.tif"
shp_path = r'D:/AndreJuarez/Code_OilSpill/Apis/last_dance/DEPARTAMENTOS.shp'

plot_sar_with_spill_and_patches(img_path,shp_path,df)
