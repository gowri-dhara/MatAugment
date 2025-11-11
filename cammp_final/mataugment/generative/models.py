# mataugment/generative/models.py
import torch
import torch.nn as nn

# Image GAN 
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 64, 64)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z).view(z.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 64, 64)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)
        return self.model(flat)


class SimpleGAN:
    def __init__(self, latent_dim=100, img_shape=(1, 64, 64), device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(latent_dim, img_shape).to(self.device)
        self.discriminator = Discriminator(img_shape).to(self.device)
        self.latent_dim = latent_dim

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)


# Conditional GAN (labels -> one-hot condition)

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10, img_shape=(1, 64, 64)):
        super().__init__()
        self.img_shape = img_shape
        in_dim = latent_dim + n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))), nn.Tanh()
        )

    def forward(self, z, y_onehot):
        x = torch.cat([z, y_onehot], dim=1)
        img = self.net(x).view(z.size(0), *self.img_shape)
        return img


class ConditionalDiscriminator(nn.Module):
    def __init__(self, n_classes=10, img_shape=(1, 64, 64)):
        super().__init__()
        in_dim = int(torch.prod(torch.tensor(img_shape))) + n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, img, y_onehot):
        flat = img.view(img.size(0), -1)
        x = torch.cat([flat, y_onehot], dim=1)
        return self.net(x)


class ConditionalGAN:
    def __init__(self, latent_dim=100, n_classes=10, img_shape=(1, 64, 64), device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = ConditionalGenerator(latent_dim, n_classes, img_shape).to(self.device)
        self.discriminator = ConditionalDiscriminator(n_classes, img_shape).to(self.device)
        self.latent_dim = latent_dim
        self.n_classes = n_classes

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)


# DFT: Point-cloud generator (N atoms in a unit box)
# Outputs positions in [0,1]^3; atomic numbers can be provided externally.

class PointCloudGenerator(nn.Module):
    def __init__(self, latent_dim=64, n_atoms=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        out_dim = n_atoms * 3
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_dim), nn.Sigmoid()  # positions in [0,1]
        )

    def forward(self, z):
        coords = self.net(z).view(z.size(0), self.n_atoms, 3)
        return coords
