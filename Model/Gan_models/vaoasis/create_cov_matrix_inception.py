import os
import numpy as np
from torchvision.models.inception import inception_v3
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import linalg
from tqdm import tqdm

class RealSarDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted([f for f in os.listdir(root) if f.endswith('.png')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path).convert('L')  # grayscale
        img = img.resize((299, 299))
        img = img.convert('RGB')  # Inception expects 3-channel input
        if self.transform:
            img = self.transform(img)
        return img

def compute_inception_stats(dataloader, dims=2048):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    model.fc = torch.nn.Identity()  # to get 2048-dim output

    activations = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            pred = model(batch)
            activations.append(pred.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

if __name__ == '__main__':
    # Adjust these paths as needed
    image_dir = 'D:/Derrame_Data/filtered_subset_train512_gan1500/train/images'
    out_dir = 'D:/Derrame_Data/filtered_subset_train512_gan1500'
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # Normalize to [-1,1]
    ])

    dataset = RealSarDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    mu, sigma = compute_inception_stats(dataloader)
    np.save(os.path.join(out_dir, 'm.npy'), mu)
    np.save(os.path.join(out_dir, 's.npy'), sigma)

    print("\n📈 Feature Mean (mu):")
    print(f"Shape: {mu.shape}")
    print(f"First 10 values: {mu[:10]}")
    print(f"Mean of mu: {np.mean(mu):.4f}, Std of mu: {np.std(mu):.4f}")

    print("\n📊 Covariance Matrix (sigma):")
    print(f"Shape: {sigma.shape}")
    print(f"Top-left 3x3:\n{sigma[:3, :3]}")
    print(f"Trace: {np.trace(sigma):.2f}")
    print(f"Determinant: {np.linalg.det(sigma):.2e}")
    print(f"Saved mu and sigma to {out_dir}")
