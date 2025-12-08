# path: mataugment/micrograph/model.py
import torch
import torch.nn as nn

#Convolutional autoencoder for 1-channel micrographs
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 128 -> 64
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), # 64 -> 32
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),# 32 -> 16
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),# 16 -> 8
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        z = self.fc(h)
        return z

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_size=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 8 -> 16
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 16 -> 32
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32 -> 64
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # 64 -> 128
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 8, 8)
        out = self.dec(h)
        return out

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, img_size=128):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim, img_size)

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, z
