# train.py
#https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import SARLabelDataset
from bicyclegan import BiCycleGANModel
import networks
def decay_gauss_std(net):
    std = 0
    for m in net.modules():
        if isinstance(m, networks.GaussianNoise):
            m.decay_step()
            std = m.std  # Optional: track for logging
    return std
def main():
    parser = argparse.ArgumentParser()

    # Directorios y datos
    parser.add_argument('--dataroot', type=str, default='',
                        help='Folder with images/ and labels_1D/ subdirs')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--name', type=str, default='bicyclegan_experiment',
                        help='Experiment name for logging')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                        help='Preprocessing method for images')
    # Modelo y canales
    parser.add_argument('--input_nc', type=int, default=5, help='Número de canales en la entrada')
    parser.add_argument('--output_nc', type=int, default=1, help='Número de canales en la salida')
    parser.add_argument('--nz', type=int, default=64, help='Dimensión del vector latente z')

    # Redes
    parser.add_argument('--netG', type=str, default='unet_512', help='Arquitectura del generador')
    parser.add_argument('--netD', type=str, default='basic_512_multi', help='Arquitectura del discriminador')
    parser.add_argument('--netD2', type=str, default='basic_512_multi', help='selects model to use for netD2')
    parser.add_argument('--netE', type=str, default='resnet_512', help='Arquitectura del encoder')
    parser.add_argument('--nef', type=int, default=64, help='Filtros del encoder')
    parser.add_argument('--ngf', type=int, default=64, help='Filtros del generador')
    parser.add_argument('--ndf', type=int, default=64, help='Filtros del discriminador')
    parser.add_argument('--lambda_ms', type=float, default=0.05,
                            help='weight for the mode-seeking loss')
    # Hiperparámetros del entrenamiento
    parser.add_argument('--batch_size', type=int, default=1 )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')

    # Pérdidas (lambda weights)
    parser.add_argument('--lambda_GAN', type=float, default=1.0)
    parser.add_argument('--lambda_GAN2', type=float, default=1.0)
    parser.add_argument('--lambda_L1', type=float, default=10.0)
    parser.add_argument('--lambda_z', type=float, default=0.5)
    parser.add_argument('--lambda_kl', type=float, default=0.01)
    parser.add_argument('--pretrained_path_G', type=str, default=None,help='bicycle_gan/maps_net_G.pth')
    parser.add_argument('--pretrained_path_E', type=str, default=None,help = 'bicycle_gan/maps_net_E.pth')
      
    # Entrenamiento y optimización
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='[vanilla|lsgan|wgangp]')
    parser.add_argument('--norm', type=str, default='instance')
    # parser.add_argument('--use_dropout', type=bool, default=True, help='Dropout en el generador')
    parser.add_argument('--use_dropout', action='store_true', help='Enable dropout in the generator')
    parser.add_argument('--no_dropout', dest='use_dropout', action='store_false', help='Disable dropout in the generator')
    parser.set_defaults(use_dropout=True)  # default = True
    parser.add_argument('--init_gain', type=float, default=0.02)
    parser.add_argument('--init_type', type=str, default='xavier')

    # Varios
    parser.add_argument('--conditional_D', action='store_true', help='Use conditional discriminator')
    parser.add_argument('--no_conditional_D', dest='conditional_D', action='store_false', help='Use unconditional discriminator')
    parser.set_defaults(conditional_D=True)
    parser.add_argument('--num_Ds', type=int, default=2, help='Número de discriminadores')
    parser.add_argument('--direction', type=str, default='AtoB')
    parser.add_argument('--use_same_D', action='store_true', help='Share weights for discriminators')
    parser.add_argument('--no_use_same_D', dest='use_same_D', action='store_false', help='Do not share weights for discriminators')
    parser.set_defaults(use_same_D=False)


    parser.add_argument('--upsample', type=str, default='basic', help='basic | bilinear')
    parser.add_argument('--nl', type=str, default='lrelu', help='non-linearity activation: relu | lrelu | elu')
    parser.add_argument('--where_add', type=str, default='all', help='input|all|middle; where to add z in the network G')
    parser.add_argument('--noise_std_D', type=float, default=0.1, help='Gaussian noise stddev for D inputs')
    parser.add_argument('--use_label_smoothing', action='store_true', help='Use label smoothing for real samples')
    parser.add_argument('--no_label_smoothing', dest='use_label_smoothing', action='store_false', help='Disable label smoothing')
    parser.set_defaults(use_label_smoothing=True)  # default = True
    opt = parser.parse_args()

    print("Using device:", opt.device)
    opt.isTrain = True
    opt.gpu_ids = [0] if opt.device == 'cuda' and torch.cuda.is_available() else []

    # 1) Create dataset
    dataset = SARLabelDataset(root_dir=opt.dataroot)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # 2) Create model
    model = BiCycleGANModel(opt)

    # 3) Train
    for epoch in range(1, opt.epochs + 1):
        epoch_losses = {}
        for i, batch in enumerate(dataloader):
            model.set_input(batch)
            model.optimize_parameters()

            losses = model.get_current_losses()
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v


        if hasattr(model, 'netD'):
            decay_gauss_std(model.netD)
        if hasattr(model, 'netD2') and model.netD2 is not None:
            decay_gauss_std(model.netD2)


        # Promediar las pérdidas del epoch
        epoch_losses = {k: v / len(dataloader) for k, v in epoch_losses.items()}

        print(f"[Epoch {epoch}/{opt.epochs}] Losses:")
        for k, v in epoch_losses.items():
            print(f" - {k}: {v:.4f}")

        # Guardar checkpoints periódicamente
        if epoch % 5 == 0 or epoch == opt.epochs:
            checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name)
            os.makedirs(checkpoint_path, exist_ok=True)

            torch.save(model.netG.state_dict(), os.path.join(checkpoint_path, f"netG_epoch_{epoch}.pth"))
            if 'D' in model.model_names:
                torch.save(model.netD.state_dict(), os.path.join(checkpoint_path, f"netD_epoch_{epoch}.pth"))
            if 'D2' in model.model_names:
                torch.save(model.netD2.state_dict(), os.path.join(checkpoint_path, f"netD2_epoch_{epoch}.pth"))
            if 'E' in model.model_names:
                torch.save(model.netE.state_dict(), os.path.join(checkpoint_path, f"netE_epoch_{epoch}.pth"))

            print(f"Epoch {epoch} checkpoints saved.")

if __name__ == "__main__":
    main()
