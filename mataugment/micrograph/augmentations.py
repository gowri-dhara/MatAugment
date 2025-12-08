# path: mataugment/micrograph/augmentations.py
import random
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Gaussian noise transform
class AddGaussianNoise(object):
    def __init__(self, std):
        self.std = std

    def __call__(self, tensor):
        return tensor + self.std * torch.randn_like(tensor)

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


# Main augmentation function
def get_augmenter(name, size=128):
    """
    Returns a torchvision transform implementing a named augmentation pipeline.
    name options:
      - "none"
      - "basic"
      - "noise"
      - "crop"
      - "combined"
    """
    base = [
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=1),
    ]

  
    # Only normalization
    if name == "none":
        return transforms.Compose(
            base
            + [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # BASIC
    if name == "basic":
        return transforms.Compose(
            base
            + [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20, interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # NOISE (basic + Gaussian noise)
    if name == "noise":
        return transforms.Compose(
            base
            + [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20, interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                AddGaussianNoise(0.05),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # CROP (random crop + basic)
    if name == "crop":
        return transforms.Compose(
            base
            + [
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # COMBINED (everything)
    if name == "combined":
        return transforms.Compose(
            base
            + [
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25, interpolation=Image.BILINEAR),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                AddGaussianNoise(0.03),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    raise ValueError(f"Unknown augmenter: {name}")
