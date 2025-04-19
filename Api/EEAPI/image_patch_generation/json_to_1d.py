import json, os
import numpy as np
from PIL import Image
from skimage.draw import polygon

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

def create_coco_png_masks(json_path, label1d_dir, color_dir):
    os.makedirs(label1d_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    data = json.load(open(json_path,'r'))
    # group annotations by image_id
    annos_by_img = {}
    for a in data['annotations']:
        annos_by_img.setdefault(a['image_id'], []).append(a)

    for img in data['images']:
        img_id   = img['id']
        fname    = img['file_name']
        h, w     = img['height'], img['width']
        base, _  = os.path.splitext(fname)

        # 1D label mask
        mask1d = np.zeros((h, w), dtype=np.uint8)

        for anno in annos_by_img.get(img_id, []):
            orig_cat = anno['category_id']
            lbl      = label_remap.get(orig_cat, 0)
            for seg in anno['segmentation']:
                xs, ys = seg[0::2], seg[1::2]
                rr, cc = polygon(ys, xs, (h, w))
                mask1d[rr, cc] = lbl

        # save 1D PNG
        mask1d_png = os.path.join(label1d_dir, f"{base}_label_1d.png")
        Image.fromarray(mask1d, mode='L').save(mask1d_png)

        # color PNG
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for l, col in dict_label_to_color_mapping.items():
            color[mask1d == l] = col
        color_png = os.path.join(color_dir, f"{base}_label_color.png")
        Image.fromarray(color, mode='RGB').save(color_png)

        print(f"✓ {base}: 1D→{mask1d_png}, color→{color_png}")

# Example usage
json_path = r"image8bit_vv_hh\archivo1.json"

label1d_dir = r"D:\AndreJuarez\Code_OilSpill\Apis\EEAPI\image_patch_generation\image8bit_vv_hh\masks_1d"
color_dir = r"D:\AndreJuarez\Code_OilSpill\Apis\EEAPI\image_patch_generation\image8bit_vv_hh\masks_color"

create_coco_png_masks(json_path, label1d_dir, color_dir)