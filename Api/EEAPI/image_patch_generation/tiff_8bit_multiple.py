import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
import re



def normalize_to_8bit_dynamic(image, percent_clip=2):
    """
    Scale based on image percentiles, avoiding extreme outliers.
    Returns 8‑bit uint image plus the (lower, upper) used.
    """
    # Mask out NaNs
    valid = image[~np.isnan(image)]
    if valid.size == 0:
        # nothing to do
        return np.zeros_like(image, dtype=np.uint8), (0, 1)
    lower = np.percentile(valid, percent_clip)
    upper = np.percentile(valid, 100 - percent_clip)
    clipped = np.clip(image, lower, upper)
    norm = (clipped - lower) / (upper - lower) * 255.0
    return norm.astype(np.uint8), (lower, upper)

def batch_normalize_images_dual_band_dynamic(
    input_dir,
    output_dir,
    index_range=(7, 31),
    percent_clip=2,
    plot=False
):
    """
    Batch normalize SAR images (VV & VH) from dB to 8‑bit using dynamic percentiles.
    """
    os.makedirs(output_dir, exist_ok=True)
    tif_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".tif") and not f.lower().endswith("_8bit.tif")
    ]
    idx_pattern = re.compile(r"^(\d+)_")

    for fname in tif_files:
        m = idx_pattern.match(fname)
        if not m:
            continue
        idx = int(m.group(1))
        if not (index_range[0] <= idx <= index_range[1]):
            continue

        in_path = os.path.join(input_dir, fname)
        with rasterio.open(in_path) as src:
            arr = src.read().astype(np.float32)
            meta = src.meta.copy()

        normalized_bands = []
        clip_ranges = []
        for b in range(arr.shape[0]):
            band = arr[b]
            norm, (lo, hi) = normalize_to_8bit_dynamic(band, percent_clip=percent_clip)
            normalized_bands.append(norm)
            clip_ranges.append((lo, hi))

        out_count = len(normalized_bands)
        out_stack = np.stack(normalized_bands, axis=0)
        meta.update({'count': out_count, 'dtype': 'uint8'})

        out_fname = fname.replace(".tif", "_8bit.tif")
        out_path = os.path.join(output_dir, out_fname)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(out_stack)

        print(f"[✔] {fname} → {out_fname}   " +
              "clips: " + ", ".join(f"[{lo:.1f},{hi:.1f}]" for lo, hi in clip_ranges))

        if plot:
            for i, norm in enumerate(normalized_bands):
                plt.figure(figsize=(5,5))
                plt.imshow(norm, cmap='gray')
                plt.title(f"{fname} — Band {i+1} (clip {clip_ranges[i]})")
                plt.axis('off')
                plt.colorbar(label="0–255")
                plt.show()
# Define directories
input_dir = r"G:\Mi unidad\GEE_img_66_512_final\7_1_2015-03-16_S1_GRD_VV.tif"
output_dir = "D:\AndreJuarez\Code_OilSpill\Apis\EEAPI\image_patch_generation\image8bit_vv_hh"

# Execute the batch normalization
batch_normalize_images_dual_band_dynamic(
    input_dir=input_dir,
    output_dir=output_dir,
    index_range=(7, 31),
    percent_clip=2,
    plot=False
)