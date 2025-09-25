import torch
import torch.nn as nn

# -------------------------
# Encoder-Decoder Autoencoder
# -------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim=128, img_size=128):
        super(Encoder, self).__init__()
        self.img_size = img_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # (B,32,64,64)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, 2, 1),  # (B,64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1), # (B,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, 4, 2, 1), # (B,128,8,8)  ← changed from 256
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)  # adjusted from 256*8*8
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, img_size=128):
        super(Decoder, self).__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)  # adjusted

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), # (B,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (B,64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # (B,32,64,64)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # (B,1,128,128)
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z).view(-1, 128, 8, 8)  # adjusted from 256
        return self.decoder(z)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128, img_size=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim, img_size)
        self.decoder = Decoder(latent_dim, img_size)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# -------------------------
# Simple Discriminator (for GAN extension)
# -------------------------

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # (B,64,64,64)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # (B,128,32,32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 4, 2, 1), # (B,128,16,16) ← was 256
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, 4, 1, 0),   # (B,1,13,13)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)
