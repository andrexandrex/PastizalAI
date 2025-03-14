# Cell 4: Define functions for Perlin noise generation and region expansion
import os
import numpy as np
import cv2  
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.segmentation import active_contour
from noise import pnoise2
from dataset import extract_connected_regions,label_to_rgb
def generate_perlin_noise_2d(height, width, scale=20.0):
    """
    Generates a 2D Perlin noise array of shape (height, width), normalized to [0,1].
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
    Applies Perlin noise expansion on a connected region.
    
    Generates a noise patch over the region's bounding box, thresholds it,
    and merges with the original region, allowing expansion only into background.
    """
    H, W = region_mask_full.shape
    coords = np.argwhere(region_mask_full == 1)
    if coords.size == 0:
        return region_mask_full
    r1, c1 = coords.min(axis=0)
    r2, c2 = coords.max(axis=0)
    region_h = r2 - r1 + 1
    region_w = c2 - c1 + 1

    noise_patch = generate_perlin_noise_2d(region_h, region_w, scale=noise_scale)
    expansion_mask_local = (noise_patch > threshold).astype(np.uint8)
    region_local = region_mask_full[r1:r2+1, c1:c2+1]
    expanded_local = np.logical_or(region_local, expansion_mask_local).astype(np.uint8)
    mask_local = mask_1d[r1:r2+1, c1:c2+1].cpu().numpy()
    
    for rr in range(region_h):
        for cc in range(region_w):
            if expanded_local[rr, cc] == 1 and region_local[rr, cc] == 0:
                if mask_local[rr, cc] != 0:
                    expanded_local[rr, cc] = 0

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

# Test the Perlin expansion on the first region for label 1 (if exists):
# Cell 5: Define the active contour (snake) deformation function

def active_contour_expand_region(region_mask_full, alpha=0.015, beta=10.0, w_line=0.0, w_edge=1.0, debug=False):
    """
    Deforms a connected region using active contour (snake) algorithm.
    
    Uses a distance transform as a pseudo-intensity image.
    """
    contours, _ = cv2.findContours(region_mask_full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return region_mask_full
    biggest_contour = max(contours, key=cv2.contourArea)
    snake_init = biggest_contour.squeeze().astype(float)
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
            max_num_iter=250,
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

# Test active contour on the first region for label 1 (if exists):


import opensimplex

def generate_opensimplex_noise_2d(height, width, scale=20.0,
                                  seed=0,
                                  octaves=1,
                                  lacunarity=2.0,
                                  persistence=0.5):
    """
    Generates a 2D OpenSimplex noise array of shape (height, width),
    normalized to [0,1].

    Parameters:
      - scale: controls the overall frequency. Larger scale => coarser features.
      - seed: random seed for opensimplex.
      - octaves: how many layers of noise to sum.
      - lacunarity: frequency multiplier per octave.
      - persistence: amplitude multiplier per octave.

    Returns: np.array of shape (height, width) with values in [0..1].
    """
    # Initialize OpenSimplex with a given seed
    opensimplex.seed(seed)

    noise_array = np.zeros((height, width), dtype=np.float32)

    # If you want multi-octave noise, we sum multiple calls with different frequencies.
    for y in range(height):
        for x in range(width):
            # For multi-octave fractal noise:
            freq = 1.0
            amp = 1.0
            total_val = 0.0
            total_amp = 0.0

            for _ in range(octaves):
                # opensimplex.noise2(xCoord, yCoord) => returns [-1..1]
                val = opensimplex.noise2(x * freq / scale, y * freq / scale)
                total_val += val * amp
                total_amp += amp

                freq *= lacunarity
                amp *= persistence

            # Average across octaves
            final_val = total_val / total_amp  # still ~ [-1..1]
            # shift/scale to [0..1]
            final_val = 0.5 * (final_val + 1.0)
            noise_array[y, x] = final_val

    return noise_array

def opensimplex_expand_region(region_mask_full,
                              mask_1d,
                              label_value,
                              scale=20.0,
                              threshold=0.5,
                              octaves=1,
                              lacunarity=2.0,
                              persistence=0.5,
                              debug=False):
    """
    Applies OpenSimplex noise expansion on a connected region.

    1) Locates bounding box of the region in 'region_mask_full'.
    2) Generates a noise patch (height=region_h, width=region_w).
    3) Thresholds it to create an expansion mask.
    4) Merges with the original region, but only where the background is 0
       (i.e., we do NOT expand into other labels).

    Returns an updated version of 'region_mask_full' with expansions.
    """

    H, W = region_mask_full.shape
    coords = np.argwhere(region_mask_full == 1)
    if coords.size == 0:
        return region_mask_full  # no pixels => nothing to expand

    # bounding box
    r1, c1 = coords.min(axis=0)
    r2, c2 = coords.max(axis=0)
    region_h = r2 - r1 + 1
    region_w = c2 - c1 + 1

    # 1) Generate noise patch
    noise_patch = generate_opensimplex_noise_2d(
        height=region_h,
        width=region_w,
        scale=scale,
        seed=0,        # or random seed if you like
        octaves=octaves,
        lacunarity=lacunarity,
        persistence=persistence
    )

    # 2) Threshold => expansion attempt
    expansion_mask_local = (noise_patch > threshold).astype(np.uint8)

    # 3) Combine with original region
    region_local = region_mask_full[r1:r2+1, c1:c2+1]
    # OR them => attempt to expand
    expanded_local = np.logical_or(region_local, expansion_mask_local).astype(np.uint8)

    # 4) Disallow expansion into other labels
    #    i.e., only expand into background=0 in 'mask_1d'.
    mask_local = mask_1d[r1:r2+1, c1:c2+1].cpu().numpy()
    for rr in range(region_h):
        for cc in range(region_w):
            if expanded_local[rr, cc] == 1 and region_local[rr, cc] == 0:
                # if it's background in region_mask_full but not background in mask_1d => revert
                if mask_local[rr, cc] != 0:
                    expanded_local[rr, cc] = 0

    # 5) Debug visualization
    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(region_local, cmap='gray')
        plt.title("Original Region (Local)")
        plt.subplot(1, 3, 2)
        plt.imshow(noise_patch, cmap='viridis')
        plt.title("OpenSimplex Noise Patch")
        plt.subplot(1, 3, 3)
        plt.imshow(expanded_local, cmap='gray')
        plt.title("Expanded Local Region")
        plt.show()

    # 6) Update region_mask_full with expanded_local
    updated_region_mask_full = region_mask_full.copy()
    updated_region_mask_full[r1:r2+1, c1:c2+1] = expanded_local
    return updated_region_mask_full