# path: mataugment/dft/model.py
import torch
import torch.nn as nn

# Simple point-cloud autoencoder (per-atom MLP encoder + global decoder)
class PointEncoder(nn.Module):
    def __init__(self, latent_dim=128, n_atoms=64):
        super().__init__()
        self.n_atoms = n_atoms
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.global_fc = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, coords):  # (B, N, 3)
        B, N, _ = coords.shape
        pts = self.point_mlp(coords.view(B*N, 3)).view(B, N, -1)
        # simple symmetric pooling (max)
        pooled, _ = torch.max(pts, dim=1)
        z = self.global_fc(pooled)
        return z

class PointDecoder(nn.Module):
    def __init__(self, latent_dim=128, n_atoms=64):
        super().__init__()
        self.n_atoms = n_atoms
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, n_atoms*3),
            nn.Sigmoid()  # positions in [0,1]
        )

    def forward(self, z):
        out = self.net(z).view(z.size(0), self.n_atoms, 3)
        return out

class PointAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, n_atoms=64):
        super().__init__()
        self.encoder = PointEncoder(latent_dim, n_atoms)
        self.decoder = PointDecoder(latent_dim, n_atoms)

    def forward(self, coords):
        z = self.encoder(coords)
        recon = self.decoder(z)
        return recon, z
