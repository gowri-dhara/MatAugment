# mataugment/generative/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def train_gan(model, dataloader, epochs=20, lr=0.0002, save_interval=5, out_dir="generated_samples", device=None):
    '''
    Train a SimpleGAN / DCGAN model and save generated samples every few epochs.
    '''
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    G, D = model.G.to(device), model.D.to(device)

    os.makedirs(out_dir, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for real_imgs in pbar:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Train D
            optD.zero_grad()
            real_labels = torch.full((batch_size,), 0.9, device=device)
            logits_real = D(real_imgs)
            loss_real = criterion(logits_real, real_labels)

            z = torch.randn(batch_size, model.latent_dim, 1, 1, device=device)
            fake_imgs = G(z).detach()
            logits_fake = D(fake_imgs)
            loss_fake = criterion(logits_fake, torch.zeros(batch_size, device=device))

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optD.step()

            # Train G 
            optG.zero_grad()
            z = torch.randn(batch_size, model.latent_dim, 1, 1, device=device)
            fake_imgs = G(z)
            logits_fake = D(fake_imgs)
            loss_G = criterion(logits_fake, torch.ones(batch_size, device=device))
            loss_G.backward()
            optG.step()

            pbar.set_postfix({"D loss": f"{loss_D.item():.4f}", "G loss": f"{loss_G.item():.4f}"})

        # Saving generated samples every few epochs
        if epoch % save_interval == 0 or epoch == epochs:
            save_generated_samples(G, model.latent_dim, epoch, out_dir, device)

    print("Training complete")

def save_generated_samples(G, latent_dim, epoch, out_dir, device):
    #Generate and save sample images from the generator.
    G.eval()
    with torch.no_grad():
        z = torch.randn(5, latent_dim, 1, 1, device=device)
        samples = G(z).cpu()
        samples = (samples * 0.5 + 0.5).clamp(0, 1)  # unnormalize

        fig, axs = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axs[i].imshow(samples[i][0], cmap="gray")
            axs[i].axis("off")
        plt.tight_layout()
        filename = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(filename)
        plt.close(fig)
    G.train()
