import json, os, rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image

PATCH_SIZE   = 512
ZOOM_FACTORS = [2, 4]        # must match the list you previewed

def crop_multiscale_geotiff(tif_path, mask_png, centroid_json, out_root):
    prefix = os.path.splitext(os.path.basename(tif_path))[0]

    # ensure folder tree
    for z in ZOOM_FACTORS:
        os.makedirs(os.path.join(out_root, f"image_z{z}"),   exist_ok=True)
        os.makedirs(os.path.join(out_root, f"label1d_z{z}"), exist_ok=True)
        os.makedirs(os.path.join(out_root, f"labelRGB_z{z}"),exist_ok=True)

    # colour table (same as before)
    cmap = {
        0:(0,0,0), 1:(0,255,255), 2:(255,0,0),
        3:(153,76,0), 4:(0,153,0)
    }
    m_arr = np.array(Image.open(mask_png).convert('L'))

    with open(centroid_json) as f: points = json.load(f)
    with rasterio.open(tif_path) as src:
        meta = src.profile.copy()

        for idx, pt in enumerate(points):
            cx, cy = pt['x'], pt['y']
            for z in ZOOM_FACTORS:
                win_size = PATCH_SIZE * z
                half     = win_size // 2
                x0 = max(0, cx - half); y0 = max(0, cy - half)

                # ---------- GeoTIFF ----------
                win = Window(x0, y0, win_size, win_size)
                data = src.read(window=win)
                out_meta = meta.copy()
                out_meta.update({
                    'height': win_size, 'width': win_size,
                    'transform': rasterio.windows.transform(win, src.transform)
                })
                t_out = os.path.join(out_root, f"image_z{z}",
                                     f"{prefix}_c{idx}_z{z}.tif")
                with rasterio.open(t_out, 'w', **out_meta) as dst:
                    dst.write(data)

                # ---------- 1‑D label ----------
                m_patch = m_arr[y0:y0+win_size, x0:x0+win_size]
                l_out = os.path.join(out_root, f"label1d_z{z}",
                                     f"{prefix}_c{idx}_z{z}.png")
                Image.fromarray(m_patch.astype(np.uint8),'L').save(l_out)

                # ---------- RGB visual ----------
                rgb = np.zeros((win_size,win_size,3), np.uint8)
                for v,col in cmap.items():
                    rgb[m_patch==v]=col
                r_out = os.path.join(out_root, f"labelRGB_z{z}",
                                     f"{prefix}_c{idx}_z{z}.png")
                Image.fromarray(rgb,'RGB').save(r_out)

                print(f"[{idx}] zoom {z}× → saved")

tif_path = r"G:\Mi unidad\GEE_img_66_512_final\7_2_2015-03-25_S1_GRD_VV.tif"
mask_png = r"image8bit_vv_hh\masks_1d\7_2_2015-03-25_S1_GRD_VV_8bit_Z1_label_1d.png"
centroid_json = r"image8bit_vv_hh\centroids_patches\7_2_centroids.json"
out_root = r"image8bit_vv_hh\patches_multiscale"

crop_multiscale_geotiff(tif_path, mask_png, centroid_json, out_root)
