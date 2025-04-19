from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Path to your label 1D PNG
mask_path = r"image8bit_vv_hh/masks_1d/7_2_2015-03-25_S1_GRD_VV_8bit_Z1_label_1d.png"

# Your desired color map (0–4)
dict_label_to_color_mapping = {
    0: np.array([  0,   0,   0], dtype=np.uint8),   # background
    1: np.array([  0, 255, 255], dtype=np.uint8),   # oil spill
    2: np.array([255,   0,   0], dtype=np.uint8),   # look-alikes
    3: np.array([153,  76,   0], dtype=np.uint8),   # ships
    4: np.array([  0, 153,   0], dtype=np.uint8),   # land
}

# 1) Load as grayscale array
mask = np.array(Image.open(mask_path).convert('L'))

# 2) Compute statistics
unique_vals, counts = np.unique(mask, return_counts=True)
stats = {
    'shape': mask.shape,
    'min': int(mask.min()),
    'max': int(mask.max()),
    'mean': float(mask.mean()),
    'std': float(mask.std()),
    'unique_values': {int(val): int(count) for val, count in zip(unique_vals, counts)}
}

print("Mask Statistics:")
for k, v in stats.items():
    print(f"  {k}: {v}")

# 3) Visualize the raw mask
plt.figure(figsize=(6,6))
plt.imshow(mask, cmap='nipy_spectral', vmin=0, vmax=4)
plt.title("Visualization of 1D Label Mask")
plt.colorbar(ticks=range(5), label='Label Value')
plt.axis('off')
plt.show()

# 4) Plot histogram of label distribution
plt.figure(figsize=(6,4))
plt.hist(mask.flatten(), bins=np.arange(-0.5,5.5,1), rwidth=0.8)
plt.xticks(range(5))
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Histogram of Label Values")
plt.show()

# 5) Build and display RGB visualization using your mapping
h, w = mask.shape
rgb = np.zeros((h, w, 3), dtype=np.uint8)
for lbl, color in dict_label_to_color_mapping.items():
    rgb[mask == lbl] = color

plt.figure(figsize=(6,6))
plt.imshow(rgb)
plt.title("Colored Label Mask (RGB)")
plt.axis('off')
plt.show()
