# path: mataugment/dft/augmentations.py
import torch
import numpy as np

def jitter(coords, sigma=0.01):
    """Add small Gaussian jitter to coordinates, clip to [0,1]. coords: (N,3) tensor"""
    noise = torch.randn_like(coords) * sigma
    return (coords + noise).clamp(0.0, 1.0)

def random_rotation(coords):
    """Apply a random 3D rotation to coords (N,3). Rotations keep distances."""
    # convert to numpy for rotation
    c = coords.cpu().numpy()
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    psi = np.random.uniform(0, 2*np.pi)
    # ZYZ Euler might be an overkill; hence random rotation matrix via QR
    R, _ = np.linalg.qr(np.random.randn(3,3))
    c_rot = c @ R.T
    # re-normalize into [0,1] by centering and scaling using unit box
    c_rot = (c_rot - c_rot.min(axis=0)) / (c_rot.ptp(axis=0) + 1e-8)
    return torch.from_numpy(c_rot.astype(np.float32))

def atom_dropout(coords, p=0.1):
    """Randomly drop a fraction p of atoms"""
    mask = torch.rand(coords.size(0)) > p
    out = coords.clone()
    out[~mask] = 0.0
    return out
