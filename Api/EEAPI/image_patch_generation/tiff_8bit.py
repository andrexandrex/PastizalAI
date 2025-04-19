import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the original SAR dB image
tif_path = r"G:\Mi unidad\GEE_img_66_512_final\7_2_2015-03-25_S1_GRD_VV.tif"

# Output directory
output_dir = r"D:\AndreJuarez\Code_OilSpill\Apis\EEAPI\image_patch_generation\image_patchesv1\normalized"
os.makedirs(output_dir, exist_ok=True)

# Normalize to 8-bit range
def normalize_to_8bit(image, min_db=-30, max_db=0):
    clipped = np.clip(image, min_db, max_db)
    norm = (clipped - min_db) / (max_db - min_db) * 255.0
    return norm.astype(np.uint8)

# Open image
with rasterio.open(tif_path) as src:
    vv_band = src.read(1)  # Read VV or VH band
    print("Original VV stats: min", vv_band.min(), "max", vv_band.max())

    vv_8bit = normalize_to_8bit(vv_band)

    # Update metadata
    out_meta = src.meta.copy()
    out_meta.update({
        'dtype': 'uint8',
        'count': 1
    })

    # Output path
    base_name = os.path.basename(tif_path).replace(".tif", "_8bit.tif")
    out_path = os.path.join(output_dir, base_name)

    # Write new 8-bit image
    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(vv_8bit, 1)

    print(f"[✔] Saved normalized image to {out_path}")

    # Show it
    plt.figure(figsize=(8, 8))
    plt.imshow(vv_8bit, cmap='gray')
    plt.title("Normalized 8-bit VV Band")
    plt.axis("off")
    plt.colorbar(label="Pixel Value (0–255)")
    plt.show()