import os, json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2                 # only for fast resize; comment‑out & use PIL if you wish

# ------------------------------------------------------------
PATCH_SIZE    = 512                 # final size on disk
ZOOM_FACTORS  = [2, 4,6]              # what you want to preview/crop
LABEL_CMAP    = "nipy_spectral"     # just a quick look‑up palette
# ------------------------------------------------------------

def _extract_patch_at(img, cx, cy, zoom, patch_size=PATCH_SIZE):
    """Return a (patch_size×patch_size) array centred on (cx,cy) at the requested zoom."""
    h, w = img.shape[:2]
    win  = int(patch_size * zoom)
    half = win // 2
    x0   = max(0, cx - half);  y0 = max(0, cy - half)
    x1   = min(w, cx + half);  y1 = min(h, cy + half)
    patch = img[y0:y1, x0:x1]
    return cv2.resize(patch, (patch_size, patch_size), cv2.INTER_LINEAR)

def interactive_multi_scale_selector(mask_png, centroid_json):
    m = np.array(Image.open(mask_png).convert("L"))
    H, W = m.shape
    picked = []
    
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(m, cmap=LABEL_CMAP, vmin=0, vmax=4)
    ax.set_title("Click centres → previews appear; close when finished")

    def onclick(ev):
        if ev.xdata is None or ev.ydata is None:    # outside axes
            return
        cx, cy = int(ev.xdata), int(ev.ydata)

        # show previews --------------------------------------------------
        n = len(ZOOM_FACTORS)
        fig2, axs = plt.subplots(1, n, figsize=(4*n,4))
        if n == 1: axs = [axs]
        for i, z in enumerate(ZOOM_FACTORS):
            p = _extract_patch_at(m, cx, cy, z, PATCH_SIZE)
            axs[i].imshow(p, cmap=LABEL_CMAP, vmin=0, vmax=4)
            axs[i].set_title(f"zoom {z}×"); axs[i].axis('off')
        plt.tight_layout(); plt.pause(0.1)

        save = input(f"Save centroid ({cx},{cy})? y/n : ").strip().lower()
        if save == 'y':
            picked.append({'x': cx, 'y': cy})
            ax.add_patch(plt.Rectangle((cx-PATCH_SIZE//2, cy-PATCH_SIZE//2),
                                       PATCH_SIZE, PATCH_SIZE,
                                       edgecolor='yellow', lw=1.5, fill=False))
            fig.canvas.draw_idle()
            print("✓ saved")
        else:
            print("✗ skipped")
        plt.close(fig2)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    with open(centroid_json, 'w') as f:
        json.dump(picked, f, indent=2)
    print(f"{len(picked)} points written → {centroid_json}")

mask_png = r"image8bit_vv_hh\masks_1d\7_2_2015-03-25_S1_GRD_VV_8bit_Z1_label_1d.png"
centroid_json = r"image8bit_vv_hh\centroids_patches\7_2_centroids.json"
os.makedirs(os.path.dirname(centroid_json), exist_ok=True)

interactive_multi_scale_selector(mask_png, centroid_json)