# path: mataugment/dft/train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm import tqdm
from .dataset import DFTPointCloudDataset
from .augmentations import jitter, random_rotation, atom_dropout
from .model import PointAutoencoder
from ..utils.viz import visualize_pointcloud_numpy
import argparse

def train_point_ae(data_dir, out_dir, aug="none",
                   n_max=64, batch_size=16, epochs=50, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    ds = DFTPointCloudDataset(data_dir, n_max)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = PointAutoencoder(latent_dim=128, n_atoms=n_max).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs} [{aug}]")
        for coords in pbar:
            coords = coords.to(device)
            # apply augmentation to input only
            if aug == "jitter":
                coords_in = jitter(coords, sigma=0.02).to(device)
            elif aug == "rotation":
                coords_in = random_rotation(coords).to(device)
            elif aug == "dropout":
                coords_in = atom_dropout(coords, p=0.15).to(device)
            elif aug == "combined":
                coords_in = jitter(coords, sigma=0.01).to(device)
                coords_in = atom_dropout(coords_in, p=0.1).to(device)
            else:
                coords_in = coords

            opt.zero_grad()
            recon, _ = model(coords_in)
            loss = criterion(recon, coords)  # reconstructing original coordinates
            loss.backward()
            opt.step()
            running += loss.item() * coords.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        print(f"Epoch {ep} avg loss: {running / len(ds):.6f}")
        torch.save(model.state_dict(), os.path.join(out_dir, f"pointae_{aug}_ep{ep:03d}.pt"))

        if ep % max(1, epochs//4) == 0:
            # sample a batch and saving a visualization
            model.eval()
            with torch.no_grad():
                sample = next(iter(loader))[0:1].to(device) if False else next(iter(loader))
                coords_sample = sample[:4].cpu().numpy()
                recon, _ = model(sample.to(device))
                recon_np = recon.cpu().numpy()[:4]
                for i in range(min(2, len(recon_np))):
                    visualize_pointcloud_numpy(coords_sample[i], fname=os.path.join(out_dir, f"recon_{aug}_ep{ep:03d}_i{i}.png"))

    print("Training finished.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--aug", default="none")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train_point_ae(args.data_dir, args.out_dir, aug=args.aug, epochs=args.epochs)
