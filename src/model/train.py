import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from config import config
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DeepFakeModel

from ..collector.collector import FaceDataLoader


class DeepFakeTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = DeepFakeModel().to(self.device)

        # Setup optimizers
        self.setup_optimizers()

        # Setup loss functions
        self.setup_loss_functions()

        # Create directories for saving
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["sample_dir"], exist_ok=True)

        # Training history
        self.history = {
            "gen_loss": [],
            "disc_loss": [],
            "recon_loss": [],
            "adv_loss": [],
            "id_loss": [],
        }

        # Initialize weights
        self.apply_weights_init()

    def setup_optimizers(self):
        """Setup optimizers for generator and discriminator"""
        # Generator parameters (encoder + decoders)
        gen_params = (
            list(self.model.encoder.parameters())
            + list(self.model.decoder_A.parameters())
            + list(self.model.decoder_B.parameters())
        )

        # Discriminator parameters
        disc_params = list(self.model.discriminator_A.parameters()) + list(
            self.model.discriminator_B.parameters()
        )

        # Optimizers
        self.gen_optimizer = optim.Adam(
            gen_params, lr=self.config["gen_lr"], betas=(0.5, 0.999)
        )

        self.disc_optimizer = optim.Adam(
            disc_params, lr=self.config["disc_lr"], betas=(0.5, 0.999)
        )

        # Learning rate schedulers
        self.gen_scheduler = optim.lr_scheduler.StepLR(
            self.gen_optimizer, step_size=self.config["lr_decay_epoch"], gamma=0.5
        )
        self.disc_scheduler = optim.lr_scheduler.StepLR(
            self.disc_optimizer, step_size=self.config["lr_decay_epoch"], gamma=0.5
        )

    def setup_loss_functions(self):
        """Setup loss functions"""
        self.recon_criterion = nn.L1Loss()  # Reconstruction loss
        self.adv_criterion = nn.MSELoss()  # Adversarial loss (LSGAN)
        self.id_criterion = nn.L1Loss()  # Identity preservation loss

    def apply_weights_init(self):
        """Initialize model weights"""

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.model.apply(weights_init)
        print("Model weights initialized")

    def compute_gradient_penalty(self, real_images, fake_images, discriminator):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)

        # Interpolate between real and fake images
        interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(
            True
        )
        disc_interpolates = discriminator(interpolates)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()

        gen_losses = []
        disc_losses = []
        recon_losses = []
        adv_losses = []
        id_losses = []

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            real_A = batch["person_a"].to(self.device)
            real_B = batch["person_b"].to(self.device)
            batch_size = real_A.size(0)

            # Create labels for adversarial loss
            real_labels = torch.ones(batch_size, 1, 8, 8).to(
                self.device
            )  # Adjust spatial dim based on discriminator output
            fake_labels = torch.zeros(batch_size, 1, 8, 8).to(self.device)

            # ==================== Train Discriminators ====================
            self.disc_optimizer.zero_grad()

            # Encode images
            with torch.no_grad():
                encoded_A = self.model.encode(real_A)
                encoded_B = self.model.encode(real_B)

            # Generate fake images
            fake_A = self.model.decode_A(encoded_B)  # B -> A
            fake_B = self.model.decode_B(encoded_A)  # A -> B

            # Discriminator A loss
            disc_A_real = self.model.discriminate_A(real_A)
            disc_A_fake = self.model.discriminate_A(fake_A.detach())

            disc_A_loss_real = self.adv_criterion(disc_A_real, real_labels)
            disc_A_loss_fake = self.adv_criterion(disc_A_fake, fake_labels)
            disc_A_loss = (disc_A_loss_real + disc_A_loss_fake) * 0.5

            # Discriminator B loss
            disc_B_real = self.model.discriminate_B(real_B)
            disc_B_fake = self.model.discriminate_B(fake_B.detach())

            disc_B_loss_real = self.adv_criterion(disc_B_real, real_labels)
            disc_B_loss_fake = self.adv_criterion(disc_B_fake, fake_labels)
            disc_B_loss = (disc_B_loss_real + disc_B_loss_fake) * 0.5

            # Total discriminator loss
            disc_loss = disc_A_loss + disc_B_loss

            # Add gradient penalty
            if self.config["use_gradient_penalty"]:
                gp_A = self.compute_gradient_penalty(
                    real_A, fake_A, self.model.discriminator_A
                )
                gp_B = self.compute_gradient_penalty(
                    real_B, fake_B, self.model.discriminator_B
                )
                disc_loss += self.config["lambda_gp"] * (gp_A + gp_B)

            disc_loss.backward()
            self.disc_optimizer.step()

            # ==================== Train Generators ====================
            if batch_idx % self.config["gen_update_freq"] == 0:
                self.gen_optimizer.zero_grad()

                # Encode images
                encoded_A = self.model.encode(real_A)
                encoded_B = self.model.encode(real_B)

                # Reconstruction
                recon_A = self.model.decode_A(encoded_A)
                recon_B = self.model.decode_B(encoded_B)

                # Face swapping
                fake_A = self.model.decode_A(encoded_B)  # B -> A
                fake_B = self.model.decode_B(encoded_A)  # A -> B

                # Cycle consistency (optional)
                if self.config["use_cycle_consistency"]:
                    cycle_A = self.model.decode_A(self.model.encode(fake_B))
                    cycle_B = self.model.decode_B(self.model.encode(fake_A))
                    cycle_loss = self.recon_criterion(
                        cycle_A, real_A
                    ) + self.recon_criterion(cycle_B, real_B)
                else:
                    cycle_loss = 0

                # Reconstruction loss
                recon_loss_A = self.recon_criterion(recon_A, real_A)
                recon_loss_B = self.recon_criterion(recon_B, real_B)
                recon_loss = (recon_loss_A + recon_loss_B) * self.config["lambda_recon"]

                # Adversarial loss
                disc_A_fake = self.model.discriminate_A(fake_A)
                disc_B_fake = self.model.discriminate_B(fake_B)

                adv_loss_A = self.adv_criterion(disc_A_fake, real_labels)
                adv_loss_B = self.adv_criterion(disc_B_fake, real_labels)
                adv_loss = (adv_loss_A + adv_loss_B) * self.config["lambda_adv"]

                # Identity preservation loss
                if self.config["use_identity_loss"]:
                    id_loss_A = self.id_criterion(fake_A, real_A)
                    id_loss_B = self.id_criterion(fake_B, real_B)
                    id_loss = (id_loss_A + id_loss_B) * self.config["lambda_id"]
                else:
                    id_loss = 0

                # Total generator loss
                gen_loss = recon_loss + adv_loss + id_loss + cycle_loss

                gen_loss.backward()
                self.gen_optimizer.step()

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "D_loss": f"{disc_loss.item():.4f}",
                        "G_loss": f"{gen_loss.item():.4f}",
                        "Recon": f"{recon_loss.item():.4f}",
                        "Adv": f"{adv_loss.item():.4f}",
                    }
                )

                # Store losses
                gen_losses.append(gen_loss.item())
                recon_losses.append(recon_loss.item())
                adv_losses.append(adv_loss.item())
                if self.config["use_identity_loss"]:
                    id_losses.append(id_loss.item())

            disc_losses.append(disc_loss.item())

        # Update learning rates
        self.gen_scheduler.step()
        self.disc_scheduler.step()

        # Return average losses
        return {
            "gen_loss": np.mean(gen_losses) if gen_losses else 0,
            "disc_loss": np.mean(disc_losses),
            "recon_loss": np.mean(recon_losses) if recon_losses else 0,
            "adv_loss": np.mean(adv_losses) if adv_losses else 0,
            "id_loss": np.mean(id_losses) if id_losses else 0,
        }

    def validate(self, dataloader, epoch):
        """Validate the model"""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in dataloader:
                real_A = batch["person_a"].to(self.device)
                real_B = batch["person_b"].to(self.device)

                # Encode and reconstruct
                encoded_A = self.model.encode(real_A)
                encoded_B = self.model.encode(real_B)

                recon_A = self.model.decode_A(encoded_A)
                recon_B = self.model.decode_B(encoded_B)

                # Compute reconstruction loss
                recon_loss_A = self.recon_criterion(recon_A, real_A)
                recon_loss_B = self.recon_criterion(recon_B, real_B)
                recon_loss = (recon_loss_A + recon_loss_B) * 0.5

                val_losses.append(recon_loss.item())

        return np.mean(val_losses)

    def save_samples(self, dataloader, epoch, num_samples=8):
        """Save sample images for visualization"""
        self.model.eval()

        with torch.no_grad():
            batch = next(iter(dataloader))
            real_A = batch["person_a"][:num_samples].to(self.device)
            real_B = batch["person_b"][:num_samples].to(self.device)

            # Encode
            encoded_A = self.model.encode(real_A)
            encoded_B = self.model.encode(real_B)

            # Reconstruct
            recon_A = self.model.decode_A(encoded_A)
            recon_B = self.model.decode_B(encoded_B)

            # Swap faces
            swapped_A2B = self.model.decode_B(encoded_A)  # A -> B
            swapped_B2A = self.model.decode_A(encoded_B)  # B -> A

            # Create grid
            images = []
            for i in range(num_samples):
                images.extend(
                    [
                        real_A[i],
                        recon_A[i],
                        swapped_A2B[i],
                        real_B[i],
                        recon_B[i],
                        swapped_B2A[i],
                    ]
                )

            grid = vutils.make_grid(images, nrow=6, normalize=True, scale_each=True)

            # Save image
            vutils.save_image(
                grid,
                os.path.join(
                    self.config["sample_dir"], f"samples_epoch_{epoch:04d}.png"
                ),
            )

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "gen_optimizer_state_dict": self.gen_optimizer.state_dict(),
            "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
            "gen_scheduler_state_dict": self.gen_scheduler.state_dict(),
            "disc_scheduler_state_dict": self.disc_scheduler.state_dict(),
            "history": self.history,
        }

        # Save regular checkpoint
        torch.save(
            checkpoint,
            os.path.join(
                self.config["checkpoint_dir"], f"checkpoint_epoch_{epoch:04d}.pth"
            ),
        )

        # Save best model
        if is_best:
            torch.save(
                checkpoint,
                os.path.join(self.config["checkpoint_dir"], "best_model.pth"),
            )

        # Save latest model
        torch.save(
            checkpoint, os.path.join(self.config["checkpoint_dir"], "latest_model.pth")
        )

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
        self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
        self.gen_scheduler.load_state_dict(checkpoint["gen_scheduler_state_dict"])
        self.disc_scheduler.load_state_dict(checkpoint["disc_scheduler_state_dict"])

        self.history = checkpoint["history"]

        return checkpoint["epoch"]

    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Generator and Discriminator losses
        axes[0, 0].plot(self.history["gen_loss"], label="Generator Loss")
        axes[0, 0].plot(self.history["disc_loss"], label="Discriminator Loss")
        axes[0, 0].set_title("Generator vs Discriminator Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Component losses
        axes[0, 1].plot(self.history["recon_loss"], label="Reconstruction Loss")
        axes[0, 1].plot(self.history["adv_loss"], label="Adversarial Loss")
        if self.history["id_loss"]:
            axes[0, 1].plot(self.history["id_loss"], label="Identity Loss")
        axes[0, 1].set_title("Component Losses")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rates
        gen_lrs = [param_group["lr"] for param_group in self.gen_optimizer.param_groups]
        disc_lrs = [
            param_group["lr"] for param_group in self.disc_optimizer.param_groups
        ]
        axes[1, 0].plot(gen_lrs, label="Generator LR")
        axes[1, 0].plot(disc_lrs, label="Discriminator LR")
        axes[1, 0].set_title("Learning Rates")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config["checkpoint_dir"], "training_history.png"))
        plt.close()

    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        print("Starting training...")
        start_epoch = 0
        best_val_loss = float("inf")

        # Load checkpoint if resuming
        if self.config["resume_training"]:
            checkpoint_path = os.path.join(
                self.config["checkpoint_dir"], "latest_model.pth"
            )
            if os.path.exists(checkpoint_path):
                start_epoch = self.load_checkpoint(checkpoint_path)
                print(f"Resumed training from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config["num_epochs"]):
            start_time = time.time()

            # Train for one epoch
            losses = self.train_epoch(train_loader, epoch)

            # Update history
            for key, value in losses.items():
                self.history[key].append(value)

            # Validate
            val_loss = 0
            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch)

            # Save samples
            if epoch % self.config["sample_interval"] == 0:
                self.save_samples(train_loader, epoch)

            # Save checkpoint
            if epoch % self.config["checkpoint_interval"] == 0:
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best)

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch}/{self.config['num_epochs']} "
                f"Gen: {losses['gen_loss']:.4f} "
                f"Disc: {losses['disc_loss']:.4f} "
                f"Recon: {losses['recon_loss']:.4f} "
                f"Time: {epoch_time:.2f}s"
            )

            # Plot history
            if epoch % self.config["plot_interval"] == 0:
                self.plot_training_history()


def main():
    # Create dataloaders
    train_loader, val_loader = FaceDataLoader.create_dataloaders(
        data_root=config["data_root"],
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        num_workers=4,
        augment=True,
    )

    # Initialize trainer
    trainer = DeepFakeTrainer(config)

    # Start training
    trainer.train(train_loader, val_loader)

    print("Training completed!")


if __name__ == "__main__":
    main()
