import numpy as np
import rasterio
import cv2  # OpenCV for resizing
import matplotlib.pyplot as plt

def extract_multi_scale_patch(image, zoom_factor, patch_size=512):
    """
    Extract a patch at a given zoom factor from the image, resizing it to patch_size x patch_size.
    
    Parameters:
      image (np.ndarray): 2D (or 3D if multi-band) numpy array.
      zoom_factor (float): Factor by which the window size is increased.
                             For zoom=1.0, a window of patch_size is extracted.
                             For zoom=2.0, a window of 2*patch_size is extracted, then resized.
      patch_size (int): The final output patch size (default 512).
    
    Returns:
      patch (np.ndarray): Resized patch (patch_size x patch_size).
    """
    # Determine image dimensions. For multi-band images, assume channels-first.
    if image.ndim == 3:
        # Channels-first: (bands, height, width)
        height, width = image.shape[1], image.shape[2]
    else:
        height, width = image.shape

    # Compute center of the image
    center_row, center_col = height // 2, width // 2
    
    # Determine window size at this zoom factor
    window_size = int(patch_size * zoom_factor)
    half_win = window_size // 2
    
    # Determine window bounds ensuring they remain within image bounds.
    row_min = max(0, center_row - half_win)
    row_max = min(height, center_row + half_win)
    col_min = max(0, center_col - half_win)
    col_max = min(width, center_col + half_win)
    
    # Extract the window
    if image.ndim == 3:
        # Channels-first; extract and then transpose to channels-last for cv2
        window = image[:, row_min:row_max, col_min:col_max]
        window = np.transpose(window, (1, 2, 0))
    else:
        window = image[row_min:row_max, col_min:col_max]
    
    # Resize to patch_size x patch_size. cv2.resize expects (width, height)
    patch = cv2.resize(window, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    
    # If multi-band, transpose back to channels-first
    if image.ndim == 3:
        patch = np.transpose(patch, (2, 0, 1))
    
    return patch

def display_multi_scale_patches(image, zoom_factors, patch_size=512):
    """
    For each zoom factor, extract a patch from the image and display it.
    """
    num_scales = len(zoom_factors)
    fig, axs = plt.subplots(1, num_scales, figsize=(5*num_scales, 5))
    
    if num_scales == 1:
        axs = [axs]
    
    for idx, zoom in enumerate(zoom_factors):
        patch = extract_multi_scale_patch(image, zoom, patch_size)
        # For visualization, if the patch is multi-band, display only the first band.
        if patch.ndim == 3:
            display_img = patch[0]
        else:
            display_img = patch
        
        axs[idx].imshow(display_img, cmap='gray')
        axs[idx].set_title(f"Zoom Factor: {zoom}")
        axs[idx].axis("off")
    
    plt.tight_layout()
    plt.show()

# ============================================
# Example usage:
# ============================================
# Replace this with the path to your large SAR image
image_path = r"G:\Mi unidad\oefa_img_v7_tiff\9_1_2017-07-18_S1_GRD_VVVH.tif"

with rasterio.open(image_path) as src:
    # Read the first band (or adjust if you need multi-band handling)
    img = src.read(1)  # This gives a 2D numpy array
    # Optionally, you can print metadata, etc.
    print("Image shape:", img.shape)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)

# Define the zoom factors you want to inspect
zoom_factors = [1.0, 2.0,4]

# Display the patches at different zoom factors
display_multi_scale_patches(img, zoom_factors, patch_size=512)

# ============================================
# Discussion:
# ============================================
# - If you downscale the large image (say from 4400x4400 to 2200x2200) externally,
#   then cropping a 512x512 patch will cover a larger physical area.
#
# - By generating patches at multiple zoom levels, each 512x512 patch represents
#   a different effective ground resolution.
#
# - You can use these patches to train a model that becomes robust to scale variations.
