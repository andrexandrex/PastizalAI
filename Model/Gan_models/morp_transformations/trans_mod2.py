##### transformationsv2 mod#################
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

# -------------------------------
# Helper Functions for Transformation & Placement
# -------------------------------


def transform_region(region_mask):
    """
    Rotates the region_mask by a random angle in {0, 90, 180, 270}.
    No translation is done here; that will be handled in place_region_in_mask.
    
    Returns:
      - rotated: the rotated binary region mask.
      - angle: the chosen rotation angle (for logging/debug).
    """
    region_h, region_w = region_mask.shape
    angle = random.choice([0,45, 90,135, 180,235, 270])
    if angle != 0:
        center = (region_w / 2, region_h / 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(region_mask, M_rot, (region_w, region_h), flags=cv2.INTER_NEAREST)
    else:
        rotated = region_mask.copy()
    return rotated, angle
import math

def place_region_in_mask(rotated_region, global_mask, bbox, max_shift=30, placement_attempts=10):
    """
    Places 'rotated_region' into 'global_mask' around its true centroid, 
    using random polar shifts (radius <= max_shift, random angle).
    
    Partial cropping at the boundary is allowed: if part of the region 
    goes out of [0, H) x [0, W), that portion is truncated.
    
    Parameters:
      - rotated_region: binary mask of the region (already rotated), shape (Hreg, Wreg).
      - global_mask: the full canvas (H, W) where we place the region.
      - bbox: (r1, c1, r2, c2) from the original region. We use it to get the original global centroid.
      - max_shift: maximum radius for the polar shift.
      - placement_attempts: how many times we attempt to place the region if conflict arises.
    
    Returns:
      - (new_r1, new_c1): top-left in global_mask where the region was placed (if success).
      - attempt_count: how many tries it took to place (or placement_attempts if skipped).
      - None if we fail to place after all attempts.
    """
    H, W = global_mask.shape
    region_h, region_w = rotated_region.shape
    
    # 1) Find the local centroid of 'rotated_region'
    coords = np.argwhere(rotated_region == 1)
    if len(coords) == 0:
        # No foreground pixels => nothing to place
        return None, 1
    
    local_cy, local_cx = coords.mean(axis=0)  # float
    
    # 2) Compute the original global centroid from the bounding box + local centroid
    (r1, c1, r2, c2) = bbox
    global_cy_original = r1 + local_cy
    global_cx_original = c1 + local_cx
    
    # We'll try multiple times to place the region
    for attempt in range(1, placement_attempts + 1):
        # 3) Random polar shift
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, max_shift)
        shift_y = radius * math.sin(angle)
        shift_x = radius * math.cos(angle)
        
        # The new global centroid after shift
        new_cy = global_cy_original + shift_y
        new_cx = global_cx_original + shift_x
        
        # 4) The top-left corner in the global mask if we want 
        #    local centroid to align with new_cy/new_cx
        top_left_y = int(round(new_cy - local_cy))
        top_left_x = int(round(new_cx - local_cx))
        
        # We'll create a temporary overlay to do partial cropping + conflict check
        temp_overlay = np.zeros_like(global_mask, dtype=np.uint8)
        
        # Destination region in global mask
        y1_dest = top_left_y
        x1_dest = top_left_x
        y2_dest = y1_dest + region_h
        x2_dest = x1_dest + region_w
        
        # 4a) Clip if out of bounds => partial cropping
        y1_src = 0
        x1_src = 0
        y2_src = region_h
        x2_src = region_w
        
        # Top-left clipping
        if y1_dest < 0:
            y1_src = -y1_dest
            y1_dest = 0
        if x1_dest < 0:
            x1_src = -x1_dest
            x1_dest = 0
        
        # Bottom-right clipping
        if y2_dest > H:
            diff = y2_dest - H
            y2_src -= diff
            y2_dest = H
        if x2_dest > W:
            diff = x2_dest - W
            x2_src -= diff
            x2_dest = W
        
        # If the cropped region is invalid (completely out of bounds), skip
        if (y1_src >= y2_src) or (x1_src >= x2_src):
            continue
        
        # 4b) Extract the portion of 'rotated_region' that fits
        region_crop = rotated_region[y1_src:y2_src, x1_src:x2_src]
        
        # 5) Check conflict in that area
        #    (If any pixel in region_crop overlaps a nonzero pixel in global_mask)
        existing = global_mask[y1_dest:y2_dest, x1_dest:x2_dest]
        overlap = (region_crop & (existing != 0))
        if np.any(overlap):
            # conflict => try again
            continue
        
        # 6) No conflict => place
        temp_overlay[y1_dest:y2_dest, x1_dest:x2_dest] = region_crop
        
        # Return success: We combine the new region with global_mask
        return (top_left_y, top_left_x), attempt
    
    # If we get here, we failed to place after all attempts
    return None, placement_attempts

def transform_and_place_regions_for_label(regions, label_value, global_mask, max_shift=30, placement_attempts=10, debug=False):
    """
    For a given label, processes its list of morphed regions by applying a random transformation
    (translation and rotation) and then attempting to place each region into the global mask.
    The placement is done with multiple attempts, and any conflict (even from other labels) is checked.
    
    Parameters:
      - regions: list of dictionaries. Each dict must include:
            'bbox': (r1, c1, r2, c2)
            'region_mask': binary numpy array (the morphed region, cropped to bbox)
            'area': area of the region.
            'technique': string describing the morphological deformation applied.
      - label_value: the label value to assign for these regions.
      - global_mask: numpy array (H, W) representing the canvas where regions are placed.
      - max_shift: maximum translation used in transformation and placement.
      - placement_attempts: maximum number of placement attempts per region.
      - debug: if True, shows debug images.
    
    Returns:
      - placement_log: list of log dictionaries for each region containing transformation parameters,
        placement coordinates, number of attempts, and status.
    """
    placement_log = []
    # Sort regions by descending area to give larger regions higher priority.
    regions_sorted = sorted(regions, key=lambda r: r['area'], reverse=True)
    
    for reg in regions_sorted:
        bbox = reg['bbox']
        orig_top_left = (bbox[0], bbox[1])
        region_mask = reg['region_mask']
        
        # Apply random translation and rotation.
        transformed, trans_params = transform_region(region_mask, max_shift)
        
        # Attempt placement with multiple attempts.
        placement, attempt_count = place_region_in_mask(transformed, global_mask, orig_top_left, max_shift, placement_attempts)
        
        if placement is not None:
            new_r1, new_c1 = placement
            # Place the region into the global mask.
            for coord in np.argwhere(transformed == 1):
                y = new_r1 + coord[0]
                x = new_c1 + coord[1]
                global_mask[y, x] = label_value
            status = 'placed'
        else:
            status = 'skipped'
        
        log_entry = {
            'bbox': bbox,
            'placement': placement,
            'trans_params': trans_params,
            'attempts': attempt_count,
            'status': status,
            'technique': reg.get('technique', 'None')
        }
        placement_log.append(log_entry)
        
        if debug:
            plt.figure(figsize=(7, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(region_mask, cmap='gray')
            plt.title(f"Label {label_value} Original (Morphed)")
            plt.subplot(1, 2, 2)
            plt.imshow(transformed, cmap='gray')
            plt.title(f"Label {label_value} Transformed\nAttempts: {attempt_count}, Status: {status}")
            plt.suptitle(f"Technique: {reg.get('technique', 'None')}")
            plt.show()
            
    return placement_log

# -------------------------------
# Updated Augmentation Function
# -------------------------------

def augment_mask_multiclass(new_regions, H, W, max_shift=30, placement_attempts=10, debug=False):
    """
    Given a dictionary 'new_regions' with keys as label values (1..4) and values as lists
    of region dictionaries (each containing 'bbox', 'region_mask', 'area', 'technique'),
    this function will:
      1. Flatten all regions across labels into a single list.
      2. Sort the regions by descending area (larger regions first).
      3. For each region, apply a random rotation (using transform_region) and then try to place it
         in a global mask using random polar shifts (using place_region_in_mask).
      4. Ensure that placements are non-overlapping.
      
    Parameters:
      - new_regions: dict with label keys and lists of region entries.
      - H, W: dimensions of the output mask.
      - max_shift: maximum translation (polar shift) allowed.
      - placement_attempts: number of attempts per region.
      - debug: if True, displays debug images.
      
    Returns:
      - global_mask: a numpy array (H, W) with the placed regions (background=0, others=labels).
      - technique_log: dictionary logging the transformation details per label.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Create an empty global mask with background 0.
    global_mask = np.zeros((H, W), dtype=np.int64)

    # Initialize a log dictionary for each label.
    technique_log = {1: [], 2: [], 3: [], 4: []}

    # Flatten the new_regions dict into a list of entries that include the label.
    all_regions = []
    for lbl, regions in new_regions.items():
        for reg in regions:
            reg_copy = reg.copy()
            reg_copy['label'] = lbl
            all_regions.append(reg_copy)

    # Sort all regions by descending area.
    all_regions_sorted = sorted(all_regions, key=lambda r: r['area'], reverse=True)

    # Process each region
    for reg in all_regions_sorted:
        bbox = reg['bbox']       # (r1, c1, r2, c2)
        region_mask = reg['region_mask']
        technique = reg.get('technique', 'None')
        lbl = reg['label']

        # (A) Rotate using your transform_region function.
        rotated_region, angle = transform_region(region_mask)

        # (B) Attempt to place into the global mask.
        # Here, we pass the current global_mask so that conflict check is done over all classes.
        placement, attempt_count = place_region_in_mask(
            rotated_region, global_mask, bbox,
            max_shift=max_shift, placement_attempts=placement_attempts
        )

        if placement is not None:
            top_left_y, top_left_x = placement
            # Place the region: map all foreground pixels in the rotated region to its label.
            for coord in np.argwhere(rotated_region == 1):
                y = top_left_y + coord[0]
                x = top_left_x + coord[1]
                global_mask[y, x] = lbl
            status = "placed"
        else:
            status = "skipped"

        # Log the transformation and placement for this region.
        log_msg = f"{technique}+Rot({angle})+PolarShift => Attempts={attempt_count}, Status={status}"
        technique_log[lbl].append(log_msg)

        if debug:
            # Show the morphed crop, rotated region, and placement attempt overlay.
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(region_mask, cmap='gray')
            plt.title(f"Label {lbl} - Morphed Crop\nTechnique: {technique}")
            plt.subplot(1, 3, 2)
            plt.imshow(rotated_region, cmap='gray')
            plt.title(f"Rotated {angle}°")
            plt.subplot(1, 3, 3)
            plt.imshow(global_mask, cmap='nipy_spectral')
            plt.title(f"After Placement\nAttempts: {attempt_count}, {status}")
            plt.suptitle(f"Label {lbl}")
            plt.tight_layout()
            plt.show()

    return global_mask, technique_log


import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import math
import cv2
def flatten_and_augment_all_labels(
    new_regions, 
    H, 
    W, 
    max_shift=30, 
    placement_attempts=10, 
    debug=False
):
    """
    Given a dictionary `new_regions` with keys as label values (1..4) and 
    values as lists of region dictionaries, each containing:
       {
         'bbox': (r1, c1, r2, c2),
         'region_mask': binary np.array (region crop),
         'area': int,
         'technique': str
       }
    This function:
      1) Flattens all labels into a single list.
      2) Sorts regions by descending area (priority to bigger shapes).
      3) For each region, applies a random rotation from {0,90,180,270}.
      4) Attempts to place it into a global mask with random polar shift
         around the original bounding box centroid (like your place_region_in_mask).
      5) If successful, sets those pixels to the label in the final mask
         (so label=2 => those pixels become '2' in the final).
      6) Logs debug info, including the transformation (angle) and attempts.

    Returns:
      - global_mask: final augmented mask (H, W) with integer labels {0..4}
      - technique_log: dict summarizing each region's transformations
    """

    # Initialize the final mask as all 0 (background).
    global_mask = np.zeros((H, W), dtype=np.int64)

    # Initialize log: one list per label
    technique_log = {1: [], 2: [], 3: [], 4: []}

    # -----------------------------
    # 1) Flatten all regions
    # -----------------------------
    all_regions = []
    for lbl, region_list in new_regions.items():
        for reg in region_list:
            reg_copy = reg.copy()
            reg_copy["label"] = lbl  # store the label
            all_regions.append(reg_copy)

    # -----------------------------
    # 2) Sort by descending area
    # -----------------------------
    all_regions_sorted = sorted(all_regions, key=lambda r: r["area"], reverse=True)

    # -----------------------------
    # 3) Function to do random rotation
    # -----------------------------
    def transform_region(region_mask):
        angle = random.choice([0, 90, 180, 270])
        h, w = region_mask.shape

        if angle != 0:
            # Make the output size big enough so the shape isn't clipped.
            max_dim = max(h, w)
            out_size = (max_dim, max_dim)
            center = (w / 2, h / 2)
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(region_mask, M_rot, out_size, flags=cv2.INTER_NEAREST)
        else:
            rotated = region_mask.copy()
        return rotated, angle

    # -----------------------------
    # 4) place_region_in_mask (same as your code)
    # -----------------------------
    def place_region_in_mask(rotated_region, global_mask, lbl, bbox,
                             max_shift=30, placement_attempts=10):
        """
        Places 'rotated_region' (binary) into 'global_mask' (int labels) 
        using random polar shifts around the region's original centroid.
        We do partial clipping if out-of-bounds, and assign label=lbl 
        only in the clipped area. If placed, returns (True, attempt_count). 
        Else returns (False, attempt_count).
        """
        H_, W_ = global_mask.shape
        region_h, region_w = rotated_region.shape

        # 1) local centroid
        coords = np.argwhere(rotated_region == 1)
        if len(coords) == 0:
            return False, 1  # nothing to place

        local_cy, local_cx = coords.mean(axis=0)

        # 2) original global centroid from bounding box
        (r1, c1, r2, c2) = bbox
        global_cy_original = r1 + local_cy
        global_cx_original = c1 + local_cx

        # 3) attempt multiple times
        for attempt in range(1, placement_attempts + 1):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, max_shift)
            shift_y = radius * math.sin(angle)
            shift_x = radius * math.cos(angle)

            new_cy = global_cy_original + shift_y
            new_cx = global_cx_original + shift_x

            # top-left corner
            top_left_y = int(round(new_cy - local_cy))
            top_left_x = int(round(new_cx - local_cx))

            # partial clipping
            y1_dest = top_left_y
            x1_dest = top_left_x
            y2_dest = y1_dest + region_h
            x2_dest = x1_dest + region_w

            y1_src = 0
            x1_src = 0
            y2_src = region_h
            x2_src = region_w

            # clip top-left
            if y1_dest < 0:
                y1_src = -y1_dest
                y1_dest = 0
            if x1_dest < 0:
                x1_src = -x1_dest
                x1_dest = 0

            # clip bottom-right
            if y2_dest > H_:
                diff = y2_dest - H_
                y2_src -= diff
                y2_dest = H_
            if x2_dest > W_:
                diff = x2_dest - W_
                x2_src -= diff
                x2_dest = W_

            # if invalid
            if y1_src >= y2_src or x1_src >= x2_src:
                continue

            # region crop that fits
            region_crop = rotated_region[y1_src:y2_src, x1_src:x2_src]

            # check overlap
            existing = global_mask[y1_dest:y2_dest, x1_dest:x2_dest]
            overlap = (region_crop & (existing != 0))
            if np.any(overlap):
                continue

            # 4) success => do partial assignment right here
            for row in range(region_crop.shape[0]):
                for col in range(region_crop.shape[1]):
                    if region_crop[row, col] == 1:
                        global_mask[y1_dest + row, x1_dest + col] = lbl

            return True, attempt

        # failed all attempts
        return False, placement_attempts

    # -----------------------------
    # 5) Iterate over all regions
    # -----------------------------
    for reg in all_regions_sorted:
        lbl = reg["label"]
        bbox = reg["bbox"]
        region_mask = reg["region_mask"]
        technique = reg.get("technique", "None")

        # (A) Rotate
        rotated_region, angle = transform_region(region_mask)

        # (B) Place
        placed, attempt_count = place_region_in_mask(
            rotated_region, global_mask, lbl, bbox,
            max_shift=max_shift, 
            placement_attempts=placement_attempts
        )

        if placed:
            status = "placed"
        else:
            status = "skipped"

        # log
        log_msg = f"{technique}+Rot({angle}) => Attempts={attempt_count}, Status={status}"
        technique_log[lbl].append(log_msg)

        # optional debug
        if debug:
            print(f"Label={lbl}, BBox={bbox}, {log_msg}")
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(region_mask, cmap='gray')
            axs[0].set_title(f"Original Crop (technique={technique})")
            axs[1].imshow(rotated_region, cmap='gray')
            axs[1].set_title(f"Rotated {angle}°")
            axs[2].imshow(global_mask, cmap='nipy_spectral')
            axs[2].set_title("Global Mask (so far)")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

    return global_mask, technique_log