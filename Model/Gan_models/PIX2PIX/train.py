# train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import SARLabelDataset
from pix_2_pix import Pix2PixModel
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='',
                        help='Folder with images/ and labels/ subdirs')
    parser.add_argument('--pretrained_path', type=str, default='./map2sat.pth',
                        help='Optional path to map2sat.pth or similar')
    parser.add_argument('--input_nc', type=int, default=1, help='SAR channels')
    parser.add_argument('--output_nc', type=int, default=3, help='Label channels')
    parser.add_argument('--lr_policy', type=str, default='cosine', help='step/plateau/cosine')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--netG', type=str, default='unet_256',
                        help='Specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--netD', type=str, default='n_layers',
                        help='Specify discriminator architecture [basic | n_layers | pixel]')
    parser.add_argument('--gan_mode', type=str, default='vanilla', help='[vanilla|lsgan]')
    parser.add_argument('--norm', type=str, default='batch',
                        help='Normalization layer [batch | instance | none]')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--lambda_L1', type=float, default=100.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--name', type=str, default='pix2pix_experiment',
                        help='Experiment name for logging')
    parser.add_argument('--direction', type=str, default='AtoB',
                        help='Experiment name for logging')
    
    parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                        help='Preprocessing method for images')
    parser.add_argument('--no_dropout', action='store_true', 
                        help='If set, disables dropout in generator')
    parser.add_argument('--init_gain', type=float, default=0.0002, help='Scaling factor for normal, xavier, and orthogonal initialization')
    parser.add_argument('--init_type', type=str, default='normal', help='Initialization method [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='Number of layers in discriminator if using n_layers architecture')

    opt = parser.parse_args()

    print("Using device:", opt.device)
    opt.isTrain = True
    opt.gpu_ids = [0] if opt.device == 'cuda' and torch.cuda.is_available() else []

    # 1) Create dataset
    dataset = SARLabelDataset(root_dir=opt.dataroot)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # 2) Create model
    model = Pix2PixModel(opt)

    # 3) Train
    for epoch in range(1, opt.epochs+1):
        epoch_d_loss, epoch_g_loss = 0., 0.
        for i, batch in enumerate(dataloader):
            model.set_input(batch)
            model.optimize_parameters()

            losses = model.get_current_losses()
            epoch_d_loss += losses['D_real'] + losses['D_fake']

            epoch_g_loss += losses['G_GAN'] + losses['G_L1']

        n_batch = len(dataloader)
        print(f"[Epoch {epoch}/{opt.epochs}] D_loss: {epoch_d_loss/n_batch:.4f}, G_loss: {epoch_g_loss/n_batch:.4f}")

        # occasionally save checkpoint
        if epoch % 5 == 0:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save(model.netG.state_dict(), f"PIX2PIX/checkpoints/netG_epoch{epoch}.pth")
            torch.save(model.netD.state_dict(), f"PIX2PIX/checkpoints/netD_epoch{epoch}.pth")

if __name__ == "__main__":
    main()