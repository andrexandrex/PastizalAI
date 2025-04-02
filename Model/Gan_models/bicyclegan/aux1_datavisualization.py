import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_dataset(dataset):
    pixel_values = []
    label_values = []
    img_shapes = []
    label_shapes = []
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        img = sample['B'].numpy()
        label = sample['A'].numpy()
        
        # Gather pixel values
        pixel_values.append(img)  # Keep full image shape (C, H, W)
        label_values.append(label.flatten())

        # Gather shapes
        img_shapes.append(img.shape)
        label_shapes.append(label.shape)

    # Stack into a single numpy array: (N, C, H, W)
    pixel_values = np.stack(pixel_values, axis=0)
    label_values = np.concatenate(label_values)

    # Compute mean and std for each channel
    means = pixel_values.mean(axis=(0, 2, 3))   # Mean over samples, height, width
    stds = pixel_values.std(axis=(0, 2, 3))     # Std over samples, height, width

    # Image statistics
    print(f"Image dimensions (H, W): {set(img_shapes)}")
    print(f"Label dimensions (H, W): {set(label_shapes)}")
    print(f"Image Means (per channel): {means}")
    print(f"Image Stds (per channel): {stds}")
    print(f"Label - Min: {label_values.min()}, Max: {label_values.max()}, Mean: {label_values.mean()}, Std: {label_values.std()}")

    # Plot histogram of pixel intensities for each channel
    plt.figure(figsize=(10, 5))
    colors = ['red', 'green', 'blue']
    for i in range(3):
        plt.hist(pixel_values[:, i, :, :].flatten(), bins=50, alpha=0.7, label=f'Channel {i + 1}', color=colors[i])
    
    plt.title("Pixel Value Distribution (per channel)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,default='',
                        help='Folder with images/ and labels_1D/ subdirs')
    parser.add_argument('--phase', type=str, default='test',
                        help='Dataset phase (train or test)')
    
    opt = parser.parse_args()

    from dataset import SARLabelDataset
    
    dataset = SARLabelDataset(root_dir=opt.dataroot)
    
    print(f"Dataset Size: {len(dataset)} samples")
    
    analyze_dataset(dataset)

if __name__ == "__main__":
    main()