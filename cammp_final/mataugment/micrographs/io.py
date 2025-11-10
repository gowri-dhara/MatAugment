# mataugment/micrographs/io.py

import os
import cv2
import numpy as np
from torch.utils.data import Dataset

def load_all_micrographs(folder, ext=(".png", ".jpg", ".tif")):
    '''
    Loads all micrograph images from a given folder.
    Returns a list of numpy arrays normalized to [0,1].
    '''
    images = []
    for file in os.listdir(folder):
        if file.lower().endswith(ext):
            path = os.path.join(folder, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.astype('float32') / 255.0)
    return images


def save_micrograph_batch(images, folder, prefix="aug_"):
    '''
    Save a batch of images to a folder.
    '''
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(folder, f"{prefix}{i:03d}.png")
        cv2.imwrite(path, (img * 255).astype('uint8'))


class MicrographDataset(Dataset):
    '''
    PyTorch-compatible dataset for micrograph images.
    Optionally applies an augmentation on each sample.
    '''
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".tif"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = img.astype('float32') / 255.0
        if self.transform:
            img = self.transform(img)
        return img
