# path: mataugment/micrograph/dataset.py
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MicrographRawDataset(Dataset):
    """
    Loads grayscale micrograph images.
    Returns normalized tensors in [-1, 1].
    """
    def __init__(self, folder, size=128, transform=None):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        self.size = size
        self.default_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])   # [-1,1]
        ])
        self.transform = transform or self.default_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("L")
        img_t = self.transform(img)
        return img_t, os.path.basename(p)
