# file: data/sarlabeldist_dataset.py
import os
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F_trans
from torchvision.transforms import InterpolationMode

from data.base_dataset import BaseDataset, get_params, get_transform, transform_offset
# ^ "data.base_dataset" from your SPADE repo. Adjust import paths if needed.

class SarLabelDataset(BaseDataset):
    """Offline distance-map version of your SAR dataset.
       - If `opt.add_dist` is True, we load dist .npy from (root_dir/dist/).
       - If `opt.no_instance` is True, instance=0.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        # Just in case you want to default some new flags:
        # parser.add_argument('--something_extra', type=int, default=0)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root_dir = opt.dataroot
        
        # We'll assume subfolders: images/, labels_1D/, dist/
        if opt.phase == 'train':
            phase_subdir = 'train'
        else:
            phase_subdir = 'test'  # or 'val'

        self.image_dir = os.path.join(opt.dataroot, phase_subdir, 'images')
        self.label_dir = os.path.join(opt.dataroot, phase_subdir, 'labels_1D')
        self.dist_dir  = os.path.join(opt.dataroot, phase_subdir, 'dist')

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.png')])

        if len(self.image_files) == 0 or len(self.label_files) == 0:
            raise ValueError("No images or label_1D found. Check your dataset paths.")

        # Build dist paths if add_dist is True
        if opt.add_dist:
            self.dist_files = []
            for lbl in self.label_files:
                base_name = os.path.splitext(lbl)[0]  # e.g. "filename_label_1D"
                dist_name = base_name + '.npy'
                self.dist_files.append(dist_name)
        else:
            self.dist_files = []

        self.dataset_size = len(self.image_files)
        self.num_classes = 5  # for your one-hot

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # 1) Gather file names
        img_name = self.image_files[idx]
        # Assume label has same prefix, except it has '_label_1D.png'
        label_name = img_name.replace('.png', '_label_1D.png')
        #dist_name = img_name.replace('.png', '_label_1D.png')

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        # Dist path (optional)
        #dist_path = os.path.join(self.dist_dir, label_name)
        dist_tensor = 0
        if self.opt.add_dist:
            dist_name = label_name.replace('.png', '.npy')
            dist_path = os.path.join(self.dist_dir, dist_name)

        # 2) Load images as PIL
        img_pil = Image.open(img_path).convert('L')
        label_pil = Image.open(label_path).convert('L')

        # Basic train-time data augmentation
        if self.opt.isTrain:
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

        # 3) Convert to numpy => [0..1], [0..4]
        img_arr = np.array(img_pil, dtype=np.float32)/255.0
        label_arr = np.array(label_pil, dtype=np.float32)

        # 4) One-hot encode label
        h, w = label_arr.shape
        one_hot_label = np.zeros((self.num_classes, h, w), dtype=np.float32)
        for c in range(self.num_classes):
            one_hot_label[c, :, :] = (label_arr == c).astype(np.float32)

        # 5) Convert to tensor => scale image to [-1,1]
        img_tensor = torch.from_numpy(img_arr[None, :, :])  # shape (1,H,W)
        img_tensor = img_tensor * 2.0 - 1.0
        label_tensor = torch.from_numpy(one_hot_label)       # shape (5,H,W)

        # 6) Load dist if requested
        if self.opt.add_dist and os.path.isfile(dist_path):
            dist_np = np.load(dist_path)  # shape (2,H,W)
            # optionally do same flipping/rotation? But we already applied random augmentations to label.
            # This mismatch can cause minor misalignment. Usually the offline approach is done *without* random flips
            # If you want them consistent, do *no random flipping for label*, or do "online" approach.
            dist_tensor = torch.from_numpy(dist_np)  # shape (2,H,W)
        else:
            print(f"[WARNING] Missing dist file at {dist_path}, using zeros.")
            dist_tensor = torch.zeros((2, label_tensor.shape[1], label_tensor.shape[2]), dtype=torch.float32)


        # 7) Return dictionary in typical SPADE format
        return {
            'label': label_tensor,       # (5, H, W)
            'image': img_tensor,         # (1, H, W) in [-1,1]
            'path': img_path,            # or label_path
            'dist': dist_tensor,         # (2, H, W) or 0
        }
