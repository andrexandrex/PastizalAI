import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image

class SARLabelDataset(Dataset):
    """
    Dataset for Pix2Pix model: 
      - SAR images (grayscale) in `images/`
      - Corresponding labels (segmentation maps) in `labels/`
      - File matching is based on the shared filename prefix.
    """

    def __init__(self, root_dir, phase='train', transform=None):
        super().__init__()
        self.root_dir = root_dir

        self.phase = phase
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels_1D')
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('_label_1D.png')])

        if len(self.image_files) == 0 or len(self.label_files) == 0:
            raise ValueError(f"No images/labels found in {self.image_dir} or {self.label_dir}")

        self.transform = transform
 
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = img_name.replace('.png', '_label_1D.png')  # Ensure label matches

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        # Load SAR Image (grayscale)
        img = Image.open(img_path).convert('RGB')  # Ensure grayscale
        label = Image.open(label_path).convert('L')

        img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        label = np.array(label, dtype=np.float32)  # Keep as int for class labels
        img = np.transpose(img, (2,0,1))
        #img = np.expand_dims(img, axis=0)  # Convert to (1,H,W)
        label = np.expand_dims(label, axis=0)
        label_tensor = torch.from_numpy(label)
        img_tensor = torch.from_numpy(img)  # (1,H,W)

        if self.transform:
            img_tensor = self.transform(img_tensor)
            label_tensor = self.transform(label_tensor)
        return {'A': label_tensor, 'B': img_tensor, 'A_paths': [label_path], 'B_paths': [img_path]}


