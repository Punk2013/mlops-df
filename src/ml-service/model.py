import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.down_blocks = nn.Sequential(
            *[self._make_down_block(64 * 2**i, 2 if i < 2 else 4) for i in range(3)]
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(6)])

    def _make_down_block(self, channels, stride):
        return nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(6)])

        self.up_blocks = nn.Sequential(
            *[self._make_up_block(512 // (2**i)) for i in range(3)]
        )

        self.final = nn.Conv2d(64, out_channels, 7, padding=3)

    def _make_up_block(self, channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.up_blocks(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            *self._make_disc_block(64, 128),
            *self._make_disc_block(128, 256),
            *self._make_disc_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def _make_disc_block(self, in_ch, out_ch):
        return [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        ]

    def forward(self, x):
        return self.net(x)


class DeepFakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder_A = Decoder()
        self.decoder_B = Decoder()
        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()

    def encode(self, x):
        return self.encoder(x)

    def decode_A(self, z):
        return self.decoder_A(z)

    def decode_B(self, z):
        return self.decoder_B(z)

    def discriminate_A(self, x):
        return self.discriminator_A(x)

    def discriminate_B(self, x):
        return self.discriminator_B(x)


if __name__ == "__main__":
    model = DeepFakeModel()

    x = torch.randn(1, 3, 128, 128)
    encoded = model.encode(x)

    reconstructed_A = model.decode_A(encoded)
    swapped_to_B = model.decode_B(encoded)

    print("Original shape:", x.shape)
    print("Encoded shape:", encoded.shape)
    print("Reconstructed A shape:", reconstructed_A.shape)
    print("Swapped to B shape:", swapped_to_B.shape)
