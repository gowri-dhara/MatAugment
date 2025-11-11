# mataugment/generative/train_physics.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_gan_physics(model, dataloader, physics_loss, epochs=50, lr=2e-4, mode='image'):
    '''
    mode='image':
        dataloader yields real images in [-1,1], shape (B,1,H,W)
    mode='dft':
        dataloader yields dict with:
          'coords': (B,N,3) real coords in [0,1]
          (optional) 'coords_ref': reference coords for physics (defaults to same batch)
    '''
    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    if mode == 'image':
        G, D = model.generator, model.discriminator
        optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    else:  # dft point cloud
        G = model  # model is PointCloudGenerator
        D = None
        optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        optD = None

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
          
            # IMAGE MODE (GAN + physics)
           
            if mode == 'image':
                real = batch.to(device)
                B = real.size(0)
                valid = torch.ones((B, 1), device=device)
                fake = torch.zeros((B, 1), device=device)

                # Train G
                optG.zero_grad()
                z = model.sample_noise(B)
                gen = G(z)
                adv_g = criterion(D(gen), valid)           # adversarial
                phys_g, _ = physics_loss(gen, real)        # physics
                g_loss = adv_g + phys_g
                g_loss.backward()
                optG.step()

                # Train D
                optD.zero_grad()
                d_real = criterion(D(real), valid)
                d_fake = criterion(D(gen.detach()), fake)
                d_loss = 0.5 * (d_real + d_fake)
                d_loss.backward()
                optD.step()

                pbar.set_postfix({'D': f'{d_loss.item():.3f}', 'G': f'{g_loss.item():.3f}'})

            
            # DFT MODE (physics-only; generator of coords)
           
            else:
                coords_real = batch['coords'].to(device)       # (B,N,3)
                coords_ref = batch.get('coords_ref', coords_real).to(device)

                optG.zero_grad()
                B = coords_real.size(0)
                z = torch.randn(B, G.latent_dim, device=device)
                coords_gen = G(z)                               # (B,N,3)

                phys_g, _ = physics_loss(coords_gen, coords_ref)
                g_loss = phys_g                                 # (no adversarial by default)
                g_loss.backward()
                optG.step()

                pbar.set_postfix({'G_phys': f'{g_loss.item():.3f}'})
