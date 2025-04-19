import re
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
from shapely.geometry import box, Point

def plot_sar_with_spill(image_path, shapefile_path, spills_df, show_plot=False):
    """
    Plots a SAR image with spill point(s) and administrative boundaries overlay,
    and also displays a histogram of the Band 1 values as well as NaN/valid counts.

    Parameters:
        image_path (str): Full path to the Sentinel-1 GeoTIFF image.
        shapefile_path (str): Full path to the DEPARTAMENTOS.shp file.
        spills_df (pd.DataFrame): DataFrame containing oil spill data with columns
                                  'LAT', 'LON', 'Original index', and 'REGION'.
        show_plot (bool): If True, displays plots including the histogram.
        
    Returns:
        dict: Dictionary of image and spatial details.
    """
    # --- Parse index from filename ---
    image_filename = image_path.split("/")[-1].split("\\")[-1]
    match = re.match(r"(\d+)_", image_filename)
    if not match:
        raise ValueError("Could not parse index from filename.")
    image_index = int(match.group(1))
    print(f"[INFO] Parsed index from image: {image_index}")

    # --- Convert spills_df to GeoDataFrame if needed ---
    if not isinstance(spills_df, gpd.GeoDataFrame):
        spills_df["geometry"] = spills_df.apply(lambda row: Point(row["LAT"], row["LON"]), axis=1)
        spills_gdf = gpd.GeoDataFrame(spills_df, geometry="geometry", crs="EPSG:4326")
    else:
        spills_gdf = spills_df

    # --- Filter spill points ---
    filtered_spills = spills_gdf[spills_gdf["Original index"] == image_index]
    print(f"[INFO] Found {len(filtered_spills)} spill(s) for index {image_index}")
    
    # --- Read SAR image ---
    with rasterio.open(image_path) as src:
        # Read the data without masking (so you see the raw distribution)
        data = src.read(1)
        print("Data type:", data.dtype)
        
        # --- Calculate NaN statistics ---
        total_pixels = data.size
        nan_count = np.count_nonzero(np.isnan(data))
        valid_count = total_pixels - nan_count
        print("Total pixel count:", total_pixels)
        print("Number of NaN pixels:", nan_count)
        print("Number of valid (non-NaN) pixels:", valid_count)
        
        # Show some sample pixel values (if desired)
        # Note: Since data is not masked, convert to a flattened array
        sample_values = data.flatten()[:10]
        print("Some sample values:", sample_values)
            
        # Also read an unmasked copy for plotting with 'show'
        img = data  # using the same data array for plotting
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        image_bounds = box(*bounds)
        image_gdf = gpd.GeoDataFrame({'geometry': [image_bounds]}, crs=crs)
        
        # Compute basic statistics using the raw data (these include NaN if any)
        img_mean = np.nanmean(data)
        img_min = np.nanmin(data)
        img_max = np.nanmax(data)
        img_std = np.nanstd(data)
        img_shape = data.shape
        num_bands = src.count
        
        print("[IMAGE STATS]")
        print(f" - Shape: {img_shape}")
        print(f" - Bands: {num_bands}")
        print(f" - Mean: {img_mean:.2f}")
        print(f" - Min: {img_min}")
        print(f" - Max: {img_max}")
        print(f" - Std Dev: {img_std:.2f}")
    
    # --- Load and reproject shapefile ---
    matching_row = spills_df[spills_df["Original index"] == image_index]
    if matching_row.empty:
        raise ValueError(f"No spill data found for image index {image_index}")
    target_region = matching_row.iloc[0]["REGION"]
    peru_gdf = gpd.read_file(shapefile_path)
    peru_gdf = peru_gdf[peru_gdf["NOMBDEP"] == target_region]
    print(f"[INFO] Target region: {target_region}")
    if peru_gdf.crs is None:
        peru_gdf.set_crs(epsg=4326, inplace=True)
    if peru_gdf.crs != crs:
        peru_gdf = peru_gdf.to_crs(crs)

    # --- Reproject spill points ---
    if filtered_spills.crs is None:
        filtered_spills.set_crs(epsg=4326, inplace=True)
    if filtered_spills.crs != crs:
        filtered_spills = filtered_spills.to_crs(crs)

    # --- Plot image, boundaries, spills, and histogram ---
    if show_plot:
        # Create image overlay plot
        fig, ax = plt.subplots(figsize=(10, 10))
        show(img, transform=transform, ax=ax, cmap='gray', vmin=-30, vmax=5)
        peru_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=1.5, label="Departamentos")
        image_gdf.boundary.plot(ax=ax, edgecolor='blue', linewidth=2, label="Image Extent")
        filtered_spills.plot(ax=ax, color='cyan', markersize=10, label='Spill Point(s)', zorder=5)
        ax.set_title(f'SAR Image Overlay with Spill (Index: {image_index})')
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Create histogram of Band 1 values (using all pixel values regardless of nodata)
        plt.figure(figsize=(10, 5))
        plt.hist(data.flatten(), bins=50, color='skyblue', edgecolor='black')
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.title(f'Histogram of Band 1 Values (Image Index: {image_index})')
        plt.tight_layout()
        plt.show()

    return {
        'image_index': image_index,
        'image_data': img,
        'transform': transform,
        'crs': crs,
        'bounds': bounds,
        'region_gdf': peru_gdf,
        'spill_gdf': filtered_spills
    }

# ---------------------------
# Example usage:
import pandas as pd

# Load your Excel spill dataset
df = pd.read_excel(r'D:\AndreJuarez\Code_OilSpill\Apis\last_dance\oefa_and_osig_ultimo.xlsx')
# pd.read_excel(r'D:\AndreJuarez\Code_OilSpill\Apis\last_dance\LastImages_full_oilspills_66.xlsx')
#D:\AndreJuarez\Code_OilSpill\Apis\last_dance\oefa_and_osig_ultimo.xlsx

img_path = r"G:\Mi unidad\temp_testIMG\11_1_2021-11-06_S1_GRD_VVVH.tif"
shp_path = r'D:/AndreJuarez/Code_OilSpill/Apis/last_dance/DEPARTAMENTOS.shp'

# Call the function
plot_sar_with_spill(img_path, shp_path, df, show_plot=True)
