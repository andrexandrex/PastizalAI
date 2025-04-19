import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt

# Replace this with the path to your exported image
tif_path = r"G:\Mi unidad\GEE_img_66_512_final\31_3_2018-10-31_S1_GRD_VVVH.tif"

# Open with rasterio
with rasterio.open(tif_path) as src:
    img = src.read()  # read all bands
    meta = src.meta
    bounds = src.bounds
    crs = src.crs

    print("CRS:", crs)
    print("Bounds:", bounds)
    print("Image shape (bands, height, width):", img.shape)
    print("Metadata:", meta)
    print("Unique values in Band 1:", np.unique(img[0]))
    # Statistics per band
    for i in range(img.shape[0]):
        band = img[i]
        print(f"\nBand {i + 1} statistics:")
        print(f"  Min:  {band.min()}")
        print(f"  Max:  {band.max()}")
        print(f"  Mean: {band.mean():.2f}")
        print(f"  Std:  {band.std():.2f}")
        print(f"  Shape: {band.shape}")

        # Optional: histogram
        plt.figure(figsize=(6, 4))
        plt.hist(band.flatten(), bins=256, color='gray', alpha=0.7)
        plt.title(f"Histogram of Band {i + 1}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Show the first band
    plt.figure(figsize=(8, 8))
    show(img[0], cmap='gray', title="First Band (probably VV or VH)")
    plt.show()
