# mataugment/generative/utils.py
import torch
import matplotlib.pyplot as plt
from mataugment.dft.utils import visualize_pointcloud

def generate_samples(model, n=5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "generator"):
        G = model.generator.to(device).eval()
        latent_dim = model.latent_dim
    else:
        # assume model itself is a generator (like PointCloudGenerator)
        G = model.to(device).eval()
        latent_dim = getattr(model, "latent_dim", 64)

    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        samples = G(z)
    return samples


def visualize_generated(model, n=5):
    samples = generate_samples(model, n)
    
    # CASE 1: image GAN (SimpleGAN)
    if samples.ndim == 4:  # (N, C, H, W)
        imgs = (samples * 0.5 + 0.5).clamp(0, 1)
        plt.figure(figsize=(10, 2))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(imgs[i][0], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # CASE 2: DFT point cloud generator
    elif samples.ndim == 3:  # (N, n_atoms, 3)
        for i in range(n):
            visualize_pointcloud(samples[i], title=f"Generated structure {i+1}")
