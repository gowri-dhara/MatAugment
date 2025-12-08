# path: mataugment/micrograph/evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from .dataset import MicrographRawDataset
from .model import ConvAutoencoder
from .augmentations import get_augmenter
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm

def evaluate_reconstruction(model_path, data_dir, aug_name="none", img_size=128, batch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_augmenter(aug_name, size=img_size)
    ds = MicrographRawDataset(data_dir, size=img_size, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # load model
    # latent_dim matches the trained model (default 128)
    model = ConvAutoencoder(latent_dim=128, img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mse_list = []
    ssim_list = []

    for batch, _ in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        with torch.no_grad():
            rec, _ = model(batch)
        b_np = ((batch.cpu().numpy() + 1) * 127.5).astype(np.uint8)  # B,1,H,W
        r_np = ((rec.cpu().numpy() + 1) * 127.5).astype(np.uint8)

        for i in range(b_np.shape[0]):
            img = b_np[i,0]
            recimg = r_np[i,0]
            mse_list.append(np.mean((img.astype(float)-recimg.astype(float))**2))
            try:
                ssim_list.append(ssim(img, recimg))
            except Exception:
                ssim_list.append(0.0)

    return {"MSE": float(np.mean(mse_list)), "SSIM": float(np.mean(ssim_list))}
