import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

# ---------- Parameters ----------
PATCH   = 512
STRIDE  = 512

# Your desired color map (0–4)
dict_label_to_color_mapping = {
    0: np.array([  0,   0,   0], dtype=np.uint8),   # background
    1: np.array([  0, 255, 255], dtype=np.uint8),   # oil spill
    2: np.array([255,   0,   0], dtype=np.uint8),   # look-alikes
    3: np.array([153,  76,   0], dtype=np.uint8),   # ships
    4: np.array([  0, 153,   0], dtype=np.uint8),   # land
}

# Remap JSON category_id → our label id
# JSON: 1=oil, 2=look-alikes, 3=land, 4=ships
# We want: 1→1, 2→2, 3→4 (land), 4→3 (ships)
label_remap = {1:1, 2:2, 3:4, 4:3}


def relevant_windows(mask, patch=PATCH, stride=STRIDE):
    """Return list of (x0,y0) top-left coords for any window where mask==1 or 2 exists."""
    h, w = mask.shape
    wins = []
    for y0 in range(0, h - patch + 1, stride):
        for x0 in range(0, w - patch + 1, stride):
            win = mask[y0:y0+patch, x0:x0+patch]
            if np.any(win == 1) or np.any(win == 2):
                wins.append((x0, y0))
    return wins


def extract_and_save_all_patches(
    image_tif: str,
    mask_png:  str,
    out_dir:   str,
    patch:     int = PATCH,
    stride:    int = STRIDE
):
    # derive prefix from image filename (without extension)
    prefix = os.path.splitext(os.path.basename(image_tif))[0]

    # load mask array
    mask_arr = np.array(Image.open(mask_png).convert('L'))
    windows  = relevant_windows(mask_arr, patch, stride)

    # prepare output folders
    img_dir     = os.path.join(out_dir, 'image_patches')
    lbl1d_dir   = os.path.join(out_dir, 'labels1d_patches')
    lblrgb_dir  = os.path.join(out_dir, 'labels_patches')
    for d in (img_dir, lbl1d_dir, lblrgb_dir):
        os.makedirs(d, exist_ok=True)

    # open GeoTIFF for reading
    with rasterio.open(image_tif) as src:
        meta = src.profile.copy()

        for idx, (x0, y0) in enumerate(windows):
            win = Window(x0, y0, patch, patch)
            data = src.read(window=win)  # reads all bands

            # --- 1) Save GeoTIFF patch ---
            out_meta = meta.copy()
            out_meta.update({
                "height":  patch,
                "width":   patch,
                "transform": rasterio.windows.transform(win, src.transform)
            })
            tif_path = os.path.join(img_dir, f"{prefix}_patch_{idx}.tif")
            with rasterio.open(tif_path, 'w', **out_meta) as dst:
                dst.write(data)

            # --- 2) Extract & remap 1D mask patch ---
            mask_patch = mask_arr[y0:y0+patch, x0:x0+patch]
            # vectorized remapping
            remapped = np.zeros_like(mask_patch)
            #for old_lbl, new_lbl in label_remap.items():
                #remapped[mask_patch == old_lbl] = new_lbl
            # save as 8‑bit PNG
            im1d = Image.fromarray(mask_patch.astype(np.uint8), mode='L')
            png1d_path = os.path.join(lbl1d_dir, f"{prefix}_mask1d_{idx}.png")
            im1d.save(png1d_path, format='PNG')

            # --- 3) Build & save RGB visualization ---
            rgb = np.zeros((patch, patch, 3), dtype=np.uint8)
            for lbl, color in dict_label_to_color_mapping.items():
                rgb[mask_patch == lbl] = color
            imrgb = Image.fromarray(rgb, mode='RGB')
            pngrgb_path = os.path.join(lblrgb_dir, f"{prefix}_maskRGB_{idx}.png")
            imrgb.save(pngrgb_path, format='PNG')

            print(f"[{idx}] saved →\n"
                  f"    • GeoTIFF:  {tif_path}\n"
                  f"    • Mask 1d:  {png1d_path}\n"
                  f"    • Mask RGB: {pngrgb_path}")


# ---------- Example invocation ----------
real_tif   = r"G:\Mi unidad\GEE_img_66_512_final\7_1_2015-03-16_S1_GRD_VV.tif"
mask_png   = r"image8bit_vv_hh\masks_1d\7_1_2015-03-16_S1_GRD_VV_8bit_Z1_label_1d.png"
out_folder = r"image8bit_vv_hh\patches"

extract_and_save_all_patches(real_tif, mask_png, out_folder)
