import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.segmentation import active_contour
from noise import pnoise2

# ================================
# 1. Label-to-RGB Mapping Function
# ================================
label_to_rgb = {
    0: [0, 0, 0],        # Black: Sea Surface
    1: [0, 255, 255],    # Cyan: Oil Spill
    2: [255, 0, 0],      # Red: Look-alike
    3: [153, 76, 0],     # Brown: Ship
    4: [0, 153, 0]       # Green: Land
}

def translate_1d_to_rgb(mask_1d):
    """
    Convert a 1D mask (H, W) with values {0,1,...,4} to an RGB image.
    """
    h, w = mask_1d.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, rgb in label_to_rgb.items():
        mask_rgb[mask_1d == label] = rgb
    return mask_rgb

# ====================================
# 2. Region Extraction (Connected Components)
# ====================================
def extract_connected_regions(mask_1d, label_value):
    """
    Extracts connected regions (components) for a given label_value.
    
    Returns a list of dictionaries, each containing:
      - 'bbox': (min_row, min_col, max_row, max_col)
      - 'binary_mask': a full-size binary mask for that component.
    """
    # Create a binary mask (numpy uint8) for this label:
    label_mask = (mask_1d == label_value).cpu().numpy().astype(np.uint8)
    labeled = cc_label(label_mask, connectivity=2)  # 8-connected regions
    regions_info = []
    for region in regionprops(labeled):
        r1, c1, r2, c2 = region.bbox
        region_mask_full = np.zeros_like(labeled, dtype=np.uint8)
        region_mask_full[region.coords[:, 0], region.coords[:, 1]] = 1
        regions_info.append({
            'bbox': (r1, c1, r2, c2),
            'binary_mask': region_mask_full
        })
    return regions_info

# ====================================
# 3. Perlin Noise Functions & Perlin Augmentation
# ====================================
def generate_perlin_noise_2d(height, width, scale=20.0):
    """
    Generate a 2D Perlin noise image (values normalized to [0,1]).
    """
    noise_array = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            noise_array[i, j] = pnoise2(i / scale, j / scale, octaves=1)
    noise_array -= noise_array.min()
    noise_array /= noise_array.max() + 1e-8
    return noise_array

def perlin_expand_region(region_mask_full, mask_1d, label_value, noise_scale=20.0, threshold=0.5, debug=False):
    """
    Applies Perlin noise expansion on one connected region.
    
    The region's bounding box is used to generate a Perlin noise patch.
    The patch is thresholded and merged with the original region, 
    but only expanding into background (label=0).
    
    If debug=True, plots the noise patch and region bounding box.
    """
    H, W = region_mask_full.shape
    coords = np.argwhere(region_mask_full == 1)
    if coords.size == 0:
        return region_mask_full
    r1, c1 = coords.min(axis=0)
    r2, c2 = coords.max(axis=0)
    region_h = r2 - r1 + 1
    region_w = c2 - c1 + 1

    # Generate Perlin noise for the region's bounding box.
    noise_patch = generate_perlin_noise_2d(region_h, region_w, scale=noise_scale)
    expansion_mask_local = (noise_patch > threshold).astype(np.uint8)
    region_local = region_mask_full[r1:r2+1, c1:c2+1]
    expanded_local = np.logical_or(region_local, expansion_mask_local).astype(np.uint8)
    mask_local = mask_1d[r1:r2+1, c1:c2+1].cpu().numpy()

    # Only allow expansion into the background (label 0)
    for rr in range(region_h):
        for cc in range(region_w):
            if expanded_local[rr, cc] == 1 and region_local[rr, cc] == 0:
                if mask_local[rr, cc] != 0:
                    expanded_local[rr, cc] = 0

    # Debug output:
    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(region_local, cmap='gray')
        plt.title("Original Region (Local)")
        plt.subplot(1,3,2)
        plt.imshow(noise_patch, cmap='viridis')
        plt.title("Perlin Noise Patch")
        plt.subplot(1,3,3)
        plt.imshow(expanded_local, cmap='gray')
        plt.title("Expanded Local Region")
        plt.suptitle(f"Perlin Expansion Debug (Label {label_value})")
        plt.show()

    updated_region_mask_full = region_mask_full.copy()
    updated_region_mask_full[r1:r2+1, c1:c2+1] = expanded_local
    return updated_region_mask_full

def augment_mask_perlin(mask_1d, noise_scale=20.0, threshold=0.5, debug=False):
    """
    Creates an augmented mask using only the Perlin noise expansion on regions for labels 1-4.
    Returns a torch tensor mask.
    """
    mask_np = mask_1d.cpu().numpy().copy()
    H, W = mask_np.shape
    augmented_mask_np = np.zeros((H, W), dtype=np.int64)
    augmented_mask_np[mask_np == 0] = 0  # preserve background

    for label_value in [1, 2, 3, 4]:
        regions_info = extract_connected_regions(mask_1d, label_value)
        for region_dict in regions_info:
            region_mask_full = region_dict['binary_mask']
            final_region = perlin_expand_region(
                region_mask_full,
                mask_1d,
                label_value,
                noise_scale=noise_scale,
                threshold=threshold,
                debug=debug
            )
            # Merge result without overwriting existing labels.
            overlap = (augmented_mask_np != 0) & (final_region == 1)
            final_region[overlap] = 0
            augmented_mask_np[final_region == 1] = label_value

    return torch.from_numpy(augmented_mask_np)

# ====================================
# 4. Snake (Active Contour) Functions & Augmentation
# ====================================
def active_contour_expand_region(region_mask_full, alpha=0.015, beta=10.0, w_line=0.0, w_edge=1.0, debug=False):
    """
    Applies active contour (snake) deformation on one connected region.
    
    Uses a distance transform image as input to the snake.
    
    If debug=True, shows the original contour and the snake result.
    """
    # Find external contours from the binary region mask.
    contours, _ = cv2.findContours(region_mask_full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return region_mask_full
    biggest_contour = max(contours, key=cv2.contourArea)
    snake_init = biggest_contour.squeeze().astype(float)
    
    # Create a pseudo-intensity image: distance transform from the background.
    dist_transform = cv2.distanceTransform(1 - region_mask_full.astype(np.uint8), cv2.DIST_L2, 5)
    dist_transform = dist_transform / (dist_transform.max() + 1e-8)
    
    try:
        snake = active_contour(
            dist_transform,
            snake_init,
            alpha=alpha,
            beta=beta,
            w_line=w_line,
            w_edge=w_edge,
            max_iterations=250,
            convergence=0.1
        )
    except Exception as e:
        print("Active contour failed:", e)
        return region_mask_full

    snake_int = np.round(snake).astype(int)
    snake_mask = np.zeros_like(region_mask_full, dtype=np.uint8)
    cv2.fillPoly(snake_mask, [snake_int], 1)
    
    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(region_mask_full, cmap='gray')
        plt.title("Original Region")
        plt.subplot(1,3,2)
        plt.imshow(dist_transform, cmap='jet')
        plt.title("Distance Transform")
        plt.subplot(1,3,3)
        plt.imshow(snake_mask, cmap='gray')
        plt.title("Snake Result")
        plt.suptitle("Active Contour Debug")
        plt.show()

    return snake_mask

def augment_mask_snake(mask_1d, alpha=0.015, beta=10.0, w_line=0.0, w_edge=1.0, debug=False):
    """
    Creates an augmented mask using only Snake deformation on regions for labels 1-4.
    Returns a torch tensor mask.
    """
    mask_np = mask_1d.cpu().numpy().copy()
    H, W = mask_np.shape
    augmented_mask_np = np.zeros((H, W), dtype=np.int64)
    augmented_mask_np[mask_np == 0] = 0  # preserve background

    for label_value in [1, 2, 3, 4]:
        regions_info = extract_connected_regions(mask_1d, label_value)
        for region_dict in regions_info:
            region_mask_full = region_dict['binary_mask']
            final_region = active_contour_expand_region(
                region_mask_full,
                alpha=alpha,
                beta=beta,
                w_line=w_line,
                w_edge=w_edge,
                debug=debug
            )
            # Merge result without overwriting existing labels.
            overlap = (augmented_mask_np != 0) & (final_region == 1)
            final_region[overlap] = 0
            augmented_mask_np[final_region == 1] = label_value

    return torch.from_numpy(augmented_mask_np)

# ====================================
# 5. Visualization Function for Combined Output
# ====================================
def visualize_all_augmentations(original_mask, perlin_mask, snake_mask):
    """
    Displays original, Perlin-augmented, and Snake-augmented masks side by side.
    """
    original_rgb = translate_1d_to_rgb(original_mask)
    perlin_rgb = translate_1d_to_rgb(perlin_mask)
    snake_rgb = translate_1d_to_rgb(snake_mask)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Mask")
    axes[0].axis("off")
    
    axes[1].imshow(perlin_rgb)
    axes[1].set_title("Augmented Mask (Perlin)")
    axes[1].axis("off")
    
    axes[2].imshow(snake_rgb)
    axes[2].set_title("Augmented Mask (Snake)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

# ====================================
# 6. Example Usage with Debug Options
# ====================================
if __name__ == "__main__":
    # Load a sample mask (update the path to your mask file)
    sample_mask_path = r"labels_1D/img_0002_2_label_1D.png"  # Windows path example; adjust as needed
    mask_pil = Image.open(sample_mask_path).convert('L')
    mask_1d = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
    
    # Set debug flag to True to view intermediate steps for one region.
    debug = True  # Change to True for step-by-step debug plots.

    # Create augmented masks using each technique exclusively.
    perlin_augmented_mask = augment_mask_perlin(mask_1d, noise_scale=2, threshold=0.9, debug=debug)
    snake_augmented_mask = augment_mask_snake(mask_1d, alpha=0.15, beta=10.0, w_line=1, w_edge=1.0, debug=debug)

    # Visualize the original and both augmented masks side by side.
    visualize_all_augmentations(mask_1d.cpu().numpy(),
                                perlin_augmented_mask.cpu().numpy(),
                                snake_augmented_mask.cpu().numpy())
