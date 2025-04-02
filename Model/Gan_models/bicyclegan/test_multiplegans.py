# file: test_multiple_epochs.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataset import SARLabelDataset
from bicyclegan import BiCycleGANModel
import types

def parse_epochs_list(epochs_str):
    # Parse a comma-separated string into a list of epochs (as strings)
    return [e.strip() for e in epochs_str.split(',') if e.strip() != ""]

def parse_indices_list(indices_str):
    """Parse a string like "1,5,10,15,20" into a list of integers.
       e.g. "1,5,10,15,20" -> [1,5,10,15,20]."""
    # Remove possible brackets, then split by comma
    indices_str = indices_str.replace('[','').replace(']','')
    return [int(x.strip()) for x in indices_str.split(',') if x.strip() != ""]

def load_networks_epoch_suffix(self, epoch_suffix, device_str):
    """Modified load_networks: load the checkpoints for the given epoch suffix."""
    for name in self.model_names:
        if isinstance(name, str):
            # Example: name='G' -> filename: "netG_epoch_<epoch_suffix>.pth"
            load_filename = 'net%s_epoch_%s.pth' % (name, epoch_suffix)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            if os.path.isfile(load_path):
                state_dict = torch.load(load_path, map_location=device_str)
                net.load_state_dict(state_dict, strict=False)
                print(f"=> Successfully loaded {name} from epoch {epoch_suffix}")
            else:
                print(f"=> ERROR: {load_path} does NOT exist! Check your checkpoints.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Typical arguments for testing
    parser.add_argument('--dataroot', type=str, default='',
                        help='Root folder with subfolders images/ and labels_1D/')
    parser.add_argument('--checkpoints_dir', type=str, default='G:/Mi unidad/GAN_Results_bicycleganv4_lastdance',
                        help='Directory where the .pth files are saved')
    parser.add_argument('--name', type=str, default='',
                        help='Experiment name (subfolder in checkpoints_dir)')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                        help='Preprocessing method for images')
    # Instead of a single epoch, we take a comma-separated list (e.g., "20,40,60")
    parser.add_argument('--epochs_list', type=str, default='1,25,50,75,100,125',
                        help='Comma-separated list of epochs to test (e.g., "20,40,60")')
    parser.add_argument('--num_variations', type=int, default=5,
                        help='How many random z draws for the same label.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='For testing, usually batch_size=1')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cpu or cuda')
    parser.add_argument('--output_dir', type=str, default='test_results_bicyclegan_1500v3',
                        help='Directory to save test outputs')
    parser.add_argument('--num_test', type=int, default=10,
                        help='Number of test examples per checkpoint (if -1, use entire dataset)')
    parser.add_argument('--no_encode', action='store_true', help='Do not use encoder; use random z')
    parser.add_argument('--test_indices', type=str, default='1,5,10,15,20',
                        help='Comma-separated dataset indices to test, e.g. "[1,5,10,15,20]"')
    # Architecture arguments – must match training
    parser.add_argument('--input_nc', type=int, default=5,
                        help='Number of channels in the input (one-hot: 5 channels for 5 classes)')
    parser.add_argument('--output_nc', type=int, default=1,
                        help='Number of channels in the output (e.g., 1 for grayscale SAR)')
    parser.add_argument('--nz', type=int, default=256,
                        help='Dimension of the latent vector z')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--netG', type=str, default='unet_512',
                        help='Generator architecture')
    parser.add_argument('--netD', type=str, default='basic_512_multi',
                        help='Discriminator architecture')
    parser.add_argument('--netD2', type=str, default='basic_512_multi',
                        help='Second discriminator (if not using same_D)')
    parser.add_argument('--netE', type=str, default='resnet_512',
                        help='Encoder architecture')
    parser.add_argument('--gan_mode', type=str, default='lsgan')
    parser.add_argument('--norm', type=str, default='batch')
    parser.add_argument('--nl', type=str, default='lrelu', help='relu | lrelu | elu')
    
    # Updated boolean flags for dropout
    parser.add_argument('--use_dropout', action='store_true', help='Enable dropout in the generator')
    parser.add_argument('--no_dropout', dest='use_dropout', action='store_false', help='Disable dropout in the generator')
    parser.set_defaults(use_dropout=True)
    
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--init_gain', type=float, default=0.02)
    parser.add_argument('--where_add', type=str, default='all',
                        help='Where to add z in the network G (input|all|middle)')
    parser.add_argument('--upsample', type=str, default='basic')
    parser.add_argument('--direction', type=str, default='AtoB')
    
    # Updated boolean flags for conditional_D
    parser.add_argument('--conditional_D', action='store_true', help='Use conditional discriminator')
    parser.add_argument('--no_conditional_D', dest='conditional_D', action='store_false', help='Use unconditional discriminator')
    parser.set_defaults(conditional_D=True)
    
    parser.add_argument('--num_Ds', type=int, default=1)
    
    # Updated boolean flags for use_same_D
    parser.add_argument('--use_same_D', action='store_true', help='Share weights for discriminators')
    parser.add_argument('--no_use_same_D', dest='use_same_D', action='store_false', help='Do not share weights for discriminators')
    parser.set_defaults(use_same_D=False)
    
    parser.add_argument('--lambda_ms', type=float, default=0.1,
                        help='Weight for the mode-seeking loss')
    # Loss weights (not used in test but required to build the model)
    parser.add_argument('--lambda_GAN', type=float, default=1.0)
    parser.add_argument('--lambda_GAN2', type=float, default=1.0)
    parser.add_argument('--lambda_L1', type=float, default=10.0)
    parser.add_argument('--lambda_z', type=float, default=0.5)
    parser.add_argument('--lambda_kl', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0002)
    # Pretrained paths: set to None if training from scratch
    parser.add_argument('--pretrained_path_G', type=str, default=None)
    parser.add_argument('--pretrained_path_E', type=str, default=None)

    opt = parser.parse_args()

    opt.isTrain = False

    device_str = opt.device
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Using CPU.")
        device_str = 'cpu'
    opt.gpu_ids = [0] if device_str == 'cuda' else []

    print("=> Using device:", device_str)

    # Parse the list of epochs to test (e.g., "20,40,60")
    epochs_list = parse_epochs_list(opt.epochs_list)
    test_indices_list = parse_indices_list(opt.test_indices) # e.g. [1,5,10,15,20]

    # Set up the test dataset and dataloader
    dataset = SARLabelDataset(root_dir=opt.dataroot, phase='test')
    dataset_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    print("=> Dataset size:", len(dataset))

    # Create a grid: rows = number of epochs in epochs_list, columns = num_test examples
    num_epochs = len(epochs_list)
    num_examples = opt.num_test if opt.num_test > 0 else len(dataset)
    
    # Prepare a list to hold generated images and their label names per epoch
    # Prepare the model once
    model = BiCycleGANModel(opt)
    # Adjust the model.save_dir
    model.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    # Overwrite load_networks
    def load_networks_epoch(self, epoch_suffix):
        return load_networks_epoch_suffix(self, epoch_suffix, device_str)
    model.load_networks = types.MethodType(load_networks_epoch, model)
    
    # We'll store results in a dict: {epoch: [(label_name, np_image), ...], ...}
    results = {}

    # For each epoch, load the checkpoint, then for each index in test_indices_list, do inference
    for ep in epochs_list:
        print(f"\n=== Testing epoch {ep} ===")
        model.load_networks(ep)
        model.eval()

        epoch_results = []
        for idx in test_indices_list:
            #IN ALL THE Tests: 
            if idx < 0 or idx >= dataset_size:
                print(f"Index {idx} is out of range (0..{dataset_size-1}), skipping.")
                # you could continue or break
                continue
            # get the single sample from the dataset
            sample = dataset[idx]
            # wrap it into a batch dimension of 1
            batch_data = {}
            for k, v in sample.items():
                batch_data[k] = v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
            model.set_input(batch_data)

            # Run test
            with torch.no_grad():
                
                real_A, fake_B, real_B = model.test(z0=None, encode=True)

            # Convert the output from [-1,1] to [0,255]
            fake_np = fake_B[0].cpu().float().numpy()  # shape (C,H,W)
            fake_np = (fake_np + 1) / 2
            #fake_np = np.clip(fake_np, 0, 1)
            fake_np = (fake_np * 255).astype(np.uint8)
            fake_np = fake_np[0, :, :] if fake_np.shape[0] == 1 else np.transpose(fake_np, (1,2,0))

            # get label name from 'A_paths'
            label_path = sample['A_paths']
            if isinstance(label_path, list):
                label_path = label_path[0]
            label_filename = os.path.basename(label_path)
            label_name = os.path.splitext(label_filename)[0]

            epoch_results.append((label_name, fake_np))

        results[ep] = epoch_results
    # Now we create a figure with:
    # rows = number of epochs
    # columns = len(test_indices_list)
    num_rows = len(epochs_list)
    num_cols = len(test_indices_list)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))
    if num_rows == 1:
        axes = [axes]  # so we can index axes[r]
    if num_cols == 1:
        axes = [[ax] for ax in axes]

    # Fill in the grid
    for r, ep in enumerate(epochs_list):
        row_results = results[ep]
        # row_results is a list of (label_name, fake_np) in the order we tested
        # but we want to align them with test_indices_list
        # so we do for c, item in enumerate(row_results) if they are in the same order
        for c, (label_name, img_np) in enumerate(row_results):
            ax = axes[r][c]
            if opt.output_nc == 1:
                ax.imshow(img_np, cmap='gray')
            else:
                ax.imshow(img_np)
            ax.axis('off')
            # top row => label name
            if r == 0:
                ax.set_title(f"{label_name}", fontsize=12, weight='bold')
        # left column => epoch label
        axes[r][0].set_ylabel(f"Epoch {ep}", fontsize=14, rotation=90, labelpad=10, weight='bold')

    plt.tight_layout()
    os.makedirs(opt.output_dir, exist_ok=True)
    out_path = os.path.join(opt.output_dir, "results_grid9_V4.png")
    plt.savefig(out_path)
    print(f"=> Done! Grid saved at: {out_path}")