# path: mataugment/micrograph/train.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import MicrographRawDataset
from .augmentations import get_augmenter
from .model import ConvAutoencoder
from pathlib import Path
import argparse
from ..utils.viz import show_image_grid

def train_autoencoder(data_dir, out_dir, aug_name="none",
                      img_size=128, batch_size=16, epochs=20, lr=1e-3,
                      latent_dim=128, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    transform = get_augmenter(aug_name, size=img_size)
    ds = MicrographRawDataset(data_dir, size=img_size, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ConvAutoencoder(latent_dim=latent_dim, img_size=img_size).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs} [{aug_name}]")
        for batch, _ in pbar:
            batch = batch.to(device)
            opt.zero_grad()
            rec, _ = model(batch)
            loss = criterion(rec, batch)
            loss.backward()
            opt.step()
            running += loss.item() * batch.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        epoch_loss = running / len(ds)
        print(f"Epoch {ep} avg loss: {epoch_loss:.6f}")

        # saving checkpoints and sample a few of the reconstructions
        torch.save(model.state_dict(), os.path.join(out_dir, f"ae_{aug_name}_ep{ep:03d}.pt"))
        if ep % max(1, epochs//4) == 0:
            model.eval()
            with torch.no_grad():
                batch, names = next(iter(loader))
                batch = batch.to(device)[:8]
                rec, _ = model(batch)
                show_image_grid((batch.cpu(), rec.cpu()), fname=os.path.join(out_dir, f"recon_{aug_name}_ep{ep:03d}.png"))

    print("Training finished.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--aug", default="none")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    train_autoencoder(args.data_dir, args.out_dir, aug_name=args.aug, epochs=args.epochs, batch_size=args.batch)
