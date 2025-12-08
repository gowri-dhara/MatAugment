# path: mataugment/dft/evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from .dataset import DFTPointCloudDataset
from .model import PointAutoencoder
import os
from tqdm import tqdm

def evaluate_point_recon(model_path, data_dir, n_max=64, batch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = DFTPointCloudDataset(data_dir, n_max)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = PointAutoencoder(latent_dim=128, n_atoms=n_max).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mse_list = []
    with torch.no_grad():
        for coords in tqdm(loader, desc="Evaluating"):
            coords = coords.to(device)
            recon, _ = model(coords)
            mse = torch.mean((recon - coords)**2, dim=(1,2)).cpu().numpy()
            mse_list.extend(mse.tolist())
    return {"MSE_per_structure_mean": float(np.mean(mse_list)), "count": len(mse_list)}
