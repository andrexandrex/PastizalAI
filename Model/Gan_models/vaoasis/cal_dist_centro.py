"""
Offline distance computation for SAR label maps.
For each label mask in `labels_1D/`, this script computes a (2,H,W) offset map
(dx, dy) from each pixel to its connected-component centroid, then saves it to .npy.
"""

import os
import cv2
import numpy as np
import argparse

def compute_centroid_offsets(label_arr):
    """
    label_arr: 2D numpy array with integer classes [0..4].
    We treat each connected region of each class as a 'pseudo-instance'.
    We'll compute dx, dy offsets from each pixel to the region's centroid.
    Returns dist_arr shape = (2, H, W).
    """
    H, W = label_arr.shape
    # We'll store offsets in float
    dx = np.zeros((H, W), dtype=np.float32)
    dy = np.zeros((H, W), dtype=np.float32)

    # Unique classes in the label, excluding background if that's 0
    unique_classes = np.unique(label_arr)
    
    # We define a helper function for normalizing offsets to [-1,1].
    def normalize_offset(arr):
        max_val = np.max(np.abs(arr)) + 1e-5
        return arr / max_val

    for c in unique_classes:
        # You might want to skip c=0 if it's background, but let's allow it:
        mask_c = (label_arr == c).astype(np.uint8) * 255
        # Now we find connected components in this binary mask
        n_comps, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask_c, connectivity=4)
        # cc_labels is shape (H,W), each region labeled [0..n_comps-1]
        # centroids is shape (n_comps, 2) => (y, x)

        for comp_id in range(1, n_comps):
            # Skip comp_id=0 because it's the background in this connectedComponents call
            y_center, x_center = centroids[comp_id]
            # Indices belonging to this component:
            component_mask = (cc_labels == comp_id)
            # For each pixel in this component, compute dx, dy
            # dy = row_index - y_center, dx = col_index - x_center
            # We'll do it the pythonic way:
            ys, xs = np.where(component_mask)
            dx[ys, xs] = xs - x_center
            dy[ys, xs] = ys - y_center

    # Optionally normalize to [-1,1] for stability
    dx = normalize_offset(dx)
    dy = normalize_offset(dy)

    dist_arr = np.stack([dx, dy], axis=0)  # (2,H,W)
    return dist_arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',default = 'D:/Derrame_Data/filtered_subset_train512_gan1500/train', type=str,
        help='Root folder containing images/ and labels_1D/ subfolders.')
    parser.add_argument('--dist_dirname', type=str, default='dist',
        help='Name of the subfolder to store distance maps, e.g. "dist" => root_dir/dist/*.npy')
    args = parser.parse_args()

    label_dir = os.path.join(args.root_dir, 'labels_1D')
    dist_dir = os.path.join(args.root_dir, args.dist_dirname)
    os.makedirs(dist_dir, exist_ok=True)

    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
    if len(label_files) == 0:
        raise ValueError(f"No label_1D PNG files found in {label_dir}")

    for i, label_name in enumerate(label_files):
        label_path = os.path.join(label_dir, label_name)
        label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_img is None:
            print(f"WARNING: Could not read {label_path}, skipping.")
            continue

        # Ensure single-channel label
        if len(label_img.shape) == 3:
            label_img = label_img[..., 0]

        # label_img is shape (H,W)
        dist_arr = compute_centroid_offsets(label_img)

        # Save as .npy with the same base name, e.g. "XXX_label_1D.png" => "XXX_label_1D.npy"
        base_name = os.path.splitext(label_name)[0]
        dist_name = base_name + '.npy'
        dist_path = os.path.join(dist_dir, dist_name)
        np.save(dist_path, dist_arr)
        
        if i % 50 == 0:
            print(f"[{i}/{len(label_files)}] Saved dist => {dist_path}")

    print("=> Done computing distance maps for all label files.")

if __name__ == '__main__':
    main()
