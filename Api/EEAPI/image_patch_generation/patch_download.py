import os
import rasterio
from patch_generation import plot_sar_with_spill_and_patches
def save_patch(image_path, window, out_path):
    """
    Save a patch (window) from the image to a new GeoTIFF file.
    """
    with rasterio.open(image_path) as src:
        patch_data = src.read(window=window)
        patch_transform = rasterio.windows.transform(window, src.transform)

        meta = src.meta.copy()
        meta.update({
            "height": window.height,
            "width": window.width,
            "transform": patch_transform
        })

        with rasterio.open(out_path, "w", **meta) as dest:
            dest.write(patch_data)

def save_all_patches(image_path, patch_windows, output_dir):
    """
    Save all patches from the provided windows list to the output directory,
    with filenames like: basefilename_patch_{i}.tif
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    for idx, window in enumerate(patch_windows):
        patch_filename = f"{base_filename}_patch_{idx+1}.tif"
        out_path = os.path.join(output_dir, patch_filename)
        save_patch(image_path, window, out_path)
        print(f"Saved patch {idx+1} to {out_path}")
import pandas as pd
df = pd.read_excel(r'D:\AndreJuarez\Code_OilSpill\Apis\last_dance\oefa_and_osig_ultimo.xlsx')
#D:\AndreJuarez\Code_OilSpill\Apis\last_dance\oefa_and_osig_ultimo.xlsx
#D:\AndreJuarez\Code_OilSpill\Apis\last_dance\LastImages_512_oefa_v6.xlsx
# Paths

img_path = r"G:\Mi unidad\oefa_img_v7_tiff\9_1_2017-07-18_S1_GRD_VVVH.tif"
##"G:\Mi unidad\oefa_img_v7_tiff\9_1_2017-07-18_S1_GRD_VVVH.tif"
##D:\AndreJuarez/Code_OilSpill/Apis/last_dance/last_dance/oefa_v7_trial1_512_corrected/15_2016-08-22_image.tif
shp_path = r'D:/AndreJuarez/Code_OilSpill/Apis/last_dance/DEPARTAMENTOS.shp'

result = plot_sar_with_spill_and_patches(img_path, shp_path, df)

# Save the patches to an output directory.
output_directory = r"image_patchesv2"
save_all_patches(img_path, result['patch_windows'], output_directory)