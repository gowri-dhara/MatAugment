# path: mataugment/dft/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ase.io import read

class DFTPointCloudDataset(Dataset):
    """
    Loads CIF files and returns normalized atomic coordinates in [0,1].
    Pads to n_max atoms with zeros.
    """
    def __init__(self, folder, n_max=64):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".cif")]
        self.n_max = n_max

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        atoms = read(self.files[idx])
        coords = atoms.get_positions()  # shape (N,3)
        cell = atoms.get_cell_lengths_and_angles()[:3]
        # preventing division by zero cell lengths
        cell = np.maximum(cell, 1e-6)
        coords = coords / cell  # element-wise approx normalization 
        coords = np.clip(coords, 0.0, 1.0)

        N = coords.shape[0]
        if N >= self.n_max:
            coords = coords[:self.n_max]
        else:
            pad = np.zeros((self.n_max - N, 3), dtype=np.float32)
            coords = np.vstack([coords, pad])
        coords = torch.from_numpy(coords.astype(np.float32))
        return coords
