import re
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from shapely.geometry import box
from shapely.geometry import Point

def plot_sar_with_spill(image_path, shapefile_path, spills_df,show_plot = False):
    """
    Plots a SAR image with spill point(s) and administrative boundaries overlay.

    Parameters:
        image_path (str): Full path to the Sentinel-1 GeoTIFF image.
        shapefile_path (str): Full path to the DEPARTAMENTOS.shp file.
        spills_df (pd.DataFrame): DataFrame containing oil spill data with columns 'LAT', 'LON', and 'Original index'.

    Returns:
        None (displays a matplotlib plot)
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
        data = src.read(1, masked=True)
        print("Data type:", data.dtype)
        print("Number of non-masked pixels:", data.count())
        # To see a few values:
        print("Some sample values:", data.compressed()[:-10])
            
        img = src.read(1)
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        image_bounds = box(*bounds)
        image_gdf = gpd.GeoDataFrame({'geometry': [image_bounds]}, crs=crs)
            # Compute statistics
        img_mean = img.mean()
        img_min = img.min()
        img_max = img.max()
        img_std = img.std()
        img_shape = img.shape
        num_bands = src.count
        data_band1 = src.read(1, masked=True)
        print("DataBand1 shape:", data_band1.shape)
        print("DataBand1 fill count:", data_band1.count())  # How many non-masked pixels?
        print("DataBand1 min:", data_band1.min())
        print("DataBand1 max:", data_band1.max())

        print("[IMAGE STATS]")
        print(f" - Shape: {img_shape}")
        print(f" - Bands: {num_bands}")
        print(f" - Mean: {img_mean:.2f}")
        print(f" - Min: {img_min}")
        print(f" - Max: {img_max}")
        print(f" - Std Dev: {img_std:.2f}")
    # --- Load and reproject shapefile ---
        ### matching rows to the oil spill
    matching_row = spills_df[spills_df["Original index"] == image_index]
    if matching_row.empty:
        raise ValueError(f"No spill data found for image index {image_index}")
    target_region = matching_row.iloc[0]["REGION"]
    peru_gdf = gpd.read_file(shapefile_path)
    peru_gdf = peru_gdf[peru_gdf["NOMBDEP"] == target_region]
    print(target_region)
    if peru_gdf.crs is None:
        peru_gdf.set_crs(epsg=4326, inplace=True)
    if peru_gdf.crs != crs:
        peru_gdf = peru_gdf.to_crs(crs)

    # --- Reproject spill points ---
    if filtered_spills.crs is None:
        filtered_spills.set_crs(epsg=4326, inplace=True)
    if filtered_spills.crs != crs:
        filtered_spills = filtered_spills.to_crs(crs)

    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        show(img, transform=transform, ax=ax, cmap='gray')
        peru_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=1.5, label="Departamentos")
        image_gdf.boundary.plot(ax=ax, edgecolor='blue', linewidth=2, label="Image Extent")
        filtered_spills.plot(ax=ax, color='cyan', markersize=10, label='Spill Point(s)', zorder=5)
        ax.set_title(f'SAR Image Overlay with Spill (Index: {image_index})')
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        ax.legend()
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



import pandas as pd

# Load your Excel spill dataset
df = pd.read_excel(r'D:\AndreJuarez\Code_OilSpill\Apis\last_dance\LastImages_full_oilspills_66.xlsx')
#D:\AndreJuarez\Code_OilSpill\Apis\last_dance\oefa_and_osig_ultimo.xlsx
#D:\AndreJuarez\Code_OilSpill\Apis\last_dance\LastImages_512_oefa_v6.xlsx
# Paths

img_path = r"G:\Mi unidad\FullOilSpills_66img_v7_tiff\7_3_2015-03-25_S1_GRD_VV.tif"
##"G:\Mi unidad\oefa_img_v7_tiff\9_1_2017-07-18_S1_GRD_VVVH.tif"
##r"G:\Mi unidad\FullOilSpills_66img_v7_tiff (1)\9_2016-01-13_S1_GRD_VV.tif"
##D:\AndreJuarez/Code_OilSpill/Apis/last_dance/last_dance/oefa_v7_trial1_512_corrected/15_2016-08-22_image.tif
shp_path = r'D:/AndreJuarez/Code_OilSpill/Apis/last_dance/DEPARTAMENTOS.shp'

# Call the function
#plot_sar_with_spill(img_path, shp_path, df,show_plot=True)
