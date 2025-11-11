# mataugment/generative/dataset.py

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import cv2
from ase.io import read


class MicrographDataset(Dataset):
    '''
    Loads micrograph images from a directory for GAN training.
    Converts to grayscale and normalizes them to [-1, 1].
    '''
    def __init__(self, root, size=64):
        self.root = root
        self.files = [os.path.join(root, f) for f in os.listdir(root)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # scale : [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("L")
        return self.transform(img)
        
# append a DFT dataset

class MicrographDataset(Dataset):
    def __init__(self, folder, size=64):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))]
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size))
        img = (img.astype(np.float32) / 127.5) - 1.0
        return torch.from_numpy(img).unsqueeze(0)

class DFTPointCloudDataset(Dataset):
    """
    Loads atomic coordinates from CIF files using ASE.
    Each .cif file is one structure -> coordinates normalized to [0,1].
    Pads or trims to a fixed number of atoms if n_atoms is given.
    """
    def __init__(self, folder, n_atoms=None):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".cif")
        ]
        self.n_atoms = n_atoms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        atoms = read(self.files[idx])              # ASE Atoms object
        coords = atoms.get_positions()             # (N, 3)
        cell = atoms.get_cell_lengths_and_angles()[:3]  # lattice vectors
        coords = coords / cell                     # normalize to [0,1]
        coords = np.clip(coords, 0, 1)

        # Trim to fixed atom count
        if self.n_atoms:
            N = coords.shape[0]
            if N > self.n_atoms:
                coords = coords[:self.n_atoms]
            elif N < self.n_atoms:
                pad = np.zeros((self.n_atoms - N, 3))
                coords = np.vstack([coords, pad])

        coords = torch.from_numpy(coords.astype(np.float32))
        return {'coords': coords}
