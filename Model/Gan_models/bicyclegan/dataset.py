import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms.functional as F_trans
from torchvision.transforms import InterpolationMode
from torchvision.transforms import InterpolationMode
class SARLabelDataset(Dataset):
    """
    Dataset for Pix2Pix model: 
      - SAR images (3-channel or 1-channel repeated) in `images/` (for training only?).
      - Corresponding labels (segmentation maps) in `labels_1D/`.
      - File matching is based on the shared filename prefix.

    If `phase=='train'`, we can apply random flips, color jitter, etc.
    If `phase=='test'`, we do minimal/no augmentation.
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
        self.num_classes = 5

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = img_name.replace('.png', '_label_1D.png')

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        # Load images as PIL Images
        img_pil = Image.open(img_path).convert('L')
        label_pil = Image.open(label_path).convert('L')

        # Potentially do random augmentations only in train phase
        if self.phase == 'train':
            if random.random() > 0.5:
                img_pil = F_trans.hflip(img_pil)
                label_pil = F_trans.hflip(label_pil)

            if random.random() > 0.5:
                img_pil = F_trans.vflip(img_pil)
                label_pil = F_trans.vflip(label_pil)

            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img_pil = F_trans.rotate(img_pil, angle, interpolation=InterpolationMode.BILINEAR)
                label_pil = F_trans.rotate(label_pil, angle, interpolation=InterpolationMode.NEAREST)

        # Convert to numpy after augmentations
        img_arr = np.array(img_pil, dtype=np.float32)/255  # normalize [0,1]
        label_arr = np.array(label_pil, dtype=np.float32)

        #4) One-hot encode label => shape (5,H,W)
        h, w = label_arr.shape
        one_hot_label = np.zeros((self.num_classes, h, w), dtype=np.float32)
        # fill it
        for c in range(self.num_classes):
            one_hot_label[c, :, :] = (label_arr == c).astype(np.float32)

        # 5) Transpose the SAR to (C,H,W), then scale to [-1,1]
        #img_arr = img_arr.transpose((2,0,1))  # => shape (3,H,W)
        #img_arr = np.stack([img_arr] * 3, axis=0)
        img_arr = img_arr[None, :, :]  # shape (1, H, W)
        img_tensor = torch.from_numpy(img_arr)
        img_tensor = (img_tensor* 2.0) - 1.0  # [-1,1]

        # 6) Convert label to tensor
        label_tensor = torch.from_numpy(one_hot_label)  # shape (5,H,W), already in [0,1]
        # (optionally apply self.transform if you want)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
            label_tensor = self.transform(label_tensor)

        return {
            'A': label_tensor,   # shape (5,H,W)
            'B': img_tensor,     # shape (3,H,W) or whatever your SAR is
            'A_paths': label_path,
            'B_paths': img_path
        }