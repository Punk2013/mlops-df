# single_person_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.down_blocks = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(512) for _ in range(6)]
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(512) for _ in range(6)]
        )

        self.up_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.up_blocks(x)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class SinglePersonDeepFake:
    def __init__(self, encoder_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize components
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Load encoder if provided
        if encoder_path and encoder_path.exists():
            self.load_encoder(encoder_path)
            print(f"Loaded encoder from {encoder_path}")
        else:
            print("Initialized with new encoder")

    def load_encoder(self, encoder_path):
        """Load encoder weights from file"""
        checkpoint = torch.load(encoder_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint)

    def save_models(self, output_dir, person_name):
        """Save all models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save encoder
        torch.save(
            {'encoder_state_dict': self.encoder.state_dict()},
            output_dir / f"encoder_{person_name}.pth"
        )

        # Save decoder
        torch.save(
            {'decoder_state_dict': self.decoder.state_dict()},
            output_dir / f"decoder_{person_name}.pth"
        )

        # Save discriminator
        torch.save(
            {'discriminator_state_dict': self.discriminator.state_dict()},
            output_dir / f"discriminator_{person_name}.pth"
        )

        print(f"Models saved to {output_dir}/")

    def load_all_models(self, model_dir, person_name):
        """Load all models for a person"""
        model_dir = Path(model_dir)

        # Load encoder
        encoder_path = model_dir / f"encoder_{person_name}.pth"
        if encoder_path.exists():
            self.load_encoder(encoder_path)

        # Load decoder
        decoder_path = model_dir / f"decoder_{person_name}.pth"
        if decoder_path.exists():
            checkpoint = torch.load(decoder_path, map_location=self.device)
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        # Load discriminator
        disc_path = model_dir / f"discriminator_{person_name}.pth"
        if disc_path.exists():
            checkpoint = torch.load(disc_path, map_location=self.device)
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
