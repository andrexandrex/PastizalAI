import os
import glob
import re
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from shapely.geometry import box, Point
import pandas as pd

def plot_sar_with_spill_details(image_path, shapefile_path, spills_df):
    """
    Process a single SAR image and associated spill data.
    Returns a dictionary with image details and geometries for plotting.
    """
    # --- Parse index from filename ---
    image_filename = os.path.basename(image_path)
    match = re.match(r"(\d+)_", image_filename)
    if not match:
        raise ValueError("Could not parse index from filename.")
    image_index = int(match.group(1))
    
    # --- Convert spills_df to GeoDataFrame if needed ---
    if not isinstance(spills_df, gpd.GeoDataFrame):
        spills_df["geometry"] = spills_df.apply(lambda row: Point(row["LAT"], row["LON"]), axis=1)
        spills_gdf = gpd.GeoDataFrame(spills_df, geometry="geometry", crs="EPSG:4326")
    else:
        spills_gdf = spills_df

    # --- Filter spill points ---
    filtered_spills = spills_gdf[spills_gdf["Original index"] == image_index]
    
    # --- Read SAR image ---
    with rasterio.open(image_path) as src:
        img = src.read(1)
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        image_bounds = box(*bounds)
        image_gdf = gpd.GeoDataFrame({'geometry': [image_bounds]}, crs=crs)
    
    # --- Load and reproject shapefile ---
    matching_row = spills_df[spills_df["Original index"] == image_index]
    if matching_row.empty:
        raise ValueError(f"No spill data found for image index {image_index}")
    target_region = matching_row.iloc[0]["REGION"]
    peru_gdf = gpd.read_file(shapefile_path)
    peru_gdf = peru_gdf[peru_gdf["NOMBDEP"] == target_region]
    if peru_gdf.crs is None:
        peru_gdf.set_crs(epsg=4326, inplace=True)
    if peru_gdf.crs != crs:
        peru_gdf = peru_gdf.to_crs(crs)
    
    # --- Reproject spill points ---
    if filtered_spills.crs is None:
        filtered_spills.set_crs(epsg=4326, inplace=True)
    if filtered_spills.crs != crs:
        filtered_spills = filtered_spills.to_crs(crs)

    return {
        'image_index': image_index,
        'image_data': img,
        'transform': transform,
        'crs': crs,
        'bounds': bounds,
        'region_gdf': peru_gdf,
        'spill_gdf': filtered_spills,
        'image_gdf': image_gdf,
        'image_name': image_filename
    }

def plot_sar_grid(folder_path, shapefile_path, spills_df, start_idx=0, end_idx=9, grid_shape=(3, 3)):
    """
    Creates a grid of SAR images with overlays for a specified range of images.
    
    Parameters:
        folder_path (str): Folder containing the GeoTIFF images.
        shapefile_path (str): Full path to the DEPARTAMENTOS.shp file.
        spills_df (pd.DataFrame): DataFrame with spill data (must contain 'LAT', 'LON', 'Original index', and 'REGION').
        start_idx (int): Starting index (inclusive) for selecting images in the sorted order.
        end_idx (int): Ending index (exclusive) for selecting images in the sorted order.
        grid_shape (tuple): Number of rows and columns for the grid.
    """
    # --- List and sort images by numeric index from the filename ---
    image_paths = glob.glob(os.path.join(folder_path, "*.tif"))
    
    def get_index(path):
        filename = os.path.basename(path)
        match = re.match(r"(\d+)_", filename)
        return int(match.group(1)) if match else float('inf')
    
    image_paths = sorted(image_paths, key=get_index)
    
    # Slice the list to only include the selected range
    selected_paths = image_paths[start_idx:end_idx]
    num_images = len(selected_paths)
    
    # --- Create the grid of subplots ---
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(15, 15))
    axes = axes.flatten()
    
    # Turn off any extra axes if there are fewer images than grid cells
    for ax in axes[num_images:]:
        ax.axis('off')
    
    # --- Loop through each selected image and plot ---
    for i, image_path in enumerate(selected_paths):
        details = plot_sar_with_spill_details(image_path, shapefile_path, spills_df)
        ax = axes[i]
        ax.axis('on')
        
        # Plot the SAR image
        show(details['image_data'], transform=details['transform'], ax=ax, cmap='gray')
        # Overlay the administrative boundaries and image extent
        details['region_gdf'].boundary.plot(ax=ax, edgecolor='red', linewidth=1.5)
        details['image_gdf'].boundary.plot(ax=ax, edgecolor='blue', linewidth=2)
        # Plot the spill point(s)
        details['spill_gdf'].plot(ax=ax, color='cyan', markersize=10, zorder=5)
        
        # Set the same extent as the image bounds
        ax.set_xlim(details['bounds'].left, details['bounds'].right)
        ax.set_ylim(details['bounds'].bottom, details['bounds'].top)
        
        # Display the image filename at the bottom of the subplot
        ax.set_title(details['image_name'], fontsize=10, pad=10)
    
    plt.tight_layout()
    plt.show()

# ---------------------------
# Example usage:
# Load the spill data from your Excel file
#df = pd.read_excel(r'D:\AndreJuarez\Code_OilSpill\Apis\last_dance\LastImages_512_oefa_v6.xlsx')
df = pd.read_excel(r'D:\AndreJuarez\Code_OilSpill\Apis\last_dance\LastImages_full_oilspills_66.xlsx')
# Set the folder containing the SAR images and the shapefile path
#folder_path = r"D:\AndreJuarez\Code_OilSpill\Apis\last_dance\last_dance\oefa_v7_trial1_512_corrected"
folder_path = r"G:\Mi unidad\FullOilSpills_66img_v7_tiff"
shp_path = r'D:/AndreJuarez/Code_OilSpill/Apis/last_dance/DEPARTAMENTOS.shp'

# Plot a grid of images from the specified range.
# For example, to plot images 10 to 18 (i.e. indices 9 through 17), use:
plot_sar_grid(folder_path, shp_path, df, start_idx=1, end_idx=6, grid_shape=(3, 3))

# Load your Excel spill dataset
#df = pd.read_excel(r'D:\AndreJuarez\Code_OilSpill\Apis\last_dance\LastImages_512_oefa_v6.xlsx')
#D:\AndreJuarez\Code_OilSpill\Apis\last_dance\oefa_and_osig_ultimo.xlsx
#D:\AndreJuarez\Code_OilSpill\Apis\last_dance\LastImages_512_oefa_v6.xlsx
# Paths

#img_path = r"D:\AndreJuarez\Code_OilSpill\Apis\last_dance\last_dance\oefa_v7_trial1_512_corrected/28_2018-06-17_image.tif"
##"G:\Mi unidad\oefa_img_v7_tiff\9_1_2017-07-18_S1_GRD_VVVH.tif"

##D:\AndreJuarez/Code_OilSpill/Apis/last_dance/last_dance/oefa_v7_trial1_512_corrected/15_2016-08-22_image.tif
#shp_path = r'D:/AndreJuarez/Code_OilSpill/Apis/last_dance/DEPARTAMENTOS.shp'
