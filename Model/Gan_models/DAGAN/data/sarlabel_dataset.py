import os
from PIL import Image
import random

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import util.util as util
from data.base_dataset import get_params, get_transform

class SarLabelDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # Start with the Pix2pixDataset options.
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        # Set defaults specific to your SAR dataset.
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=5)           # 5 classes (0,1,2,3,4)
        parser.set_defaults(aspect_ratio=1.0)         # Adjust if your patches are square
        parser.set_defaults(batchSize=8)
        return parser

    def initialize(self, opt):
        self.opt = opt
        # Get paths from our dataset folder structure.
        label_paths, image_paths, instance_paths = self.get_paths(opt)

        # Sort the paths naturally.
        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        # Limit the dataset size if opt.max_dataset_size is set.
        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        # Optionally, perform a pairing check.
        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not seem to be correctly matched. " \
                    "Please check your dataset structure or use --no_pairing_check to bypass this." \
                    % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths
        self.dataset_size = len(self.label_paths)

    def get_paths(self, opt):
        root = opt.dataroot
        # Use 'train' for training and 'test' for validation/evaluation.
        phase = 'test' if opt.phase == 'test' or opt.phase == 'val' else 'train'
        # In your SAR dataset, assume the folder structure is:
        # root/phase/images/ and root/phase/labels_1D/
        label_dir = os.path.join(root, phase, "labels_1D")
        image_dir = os.path.join(root, phase, "images")
        # Use the provided make_dataset function to list files.
        label_paths = sorted(make_dataset(label_dir, recursive=False))
        image_paths = sorted(make_dataset(image_dir, recursive=False))
        # We don't have instance maps for SAR, so we use an empty list.
        instance_paths = []  
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        # Assume path1 is the label file and path2 is the image file.
        label_name = os.path.splitext(os.path.basename(path1))[0]
        image_name = os.path.splitext(os.path.basename(path2))[0]
        # Remove the "_label_1D" suffix from the label filename.
        label_name_clean = label_name.replace("_label_1D", "")
        return label_name_clean == image_name

    def __getitem__(self, index):
        # --- Label Image ---
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        # Use nearest-neighbor for labels to avoid interpolation artifacts.
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # Scale label to [0, 255] so that the integer class values are recovered.
        label_tensor = transform_label(label) * 255.0
        # Optionally, you could set a value for 'unknown' if needed.
        #label_tensor[label_tensor == 255] = self.opt.label_nc

        # --- Input Image ---
        image_path = self.image_paths[index]
        # Check that the paths match.
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s do not match." % (label_path, image_path)
        image = Image.open(image_path).convert('L')
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        print("Image tensor shape:", image_tensor.shape)

        # --- Instance Map ---
        # Since we don't have instance maps for your SAR dataset, set instance_tensor to 0.
        instance_tensor = 0

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path}
        self.postprocess(input_dict)
        return input_dict

    def postprocess(self, input_dict):
        # Subclasses can override this to perform additional postprocessing.
        return input_dict

    def __len__(self):
        return self.dataset_size
