# single_person_trainer.py
import torch
import requests
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import numpy as np
import os
from model import SinglePersonDeepFake

class SinglePersonTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = SinglePersonDeepFake(
            encoder_path=config.get('encoder_path'),
            device=self.device
        )

        # Setup optimizers
        self.setup_optimizers()

        # Setup loss functions
        self.setup_loss_functions()

        # Freeze encoder if specified
        if config.get('freeze_encoder', False):
            self._freeze_encoder()

        # Training history
        self.history = {
            'total_loss': [],
            'recon_loss': [],
            'disc_loss': [],
            'gen_loss': []
        }

    def _freeze_encoder(self):
        """Freeze encoder parameters (only train decoder)"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen - only training decoder and discriminator")

    def setup_optimizers(self):
        """Setup optimizers for generator and discriminator"""
        # Generator parameters (encoder + decoder)
        gen_params = []
        if not self.config.get('freeze_encoder', False):
            gen_params.extend(self.model.encoder.parameters())
        gen_params.extend(self.model.decoder.parameters())

        # Discriminator parameters
        disc_params = self.model.discriminator.parameters()

        # Optimizers
        self.gen_optimizer = optim.Adam(
            gen_params,
            lr=float(self.config['gen_lr']),
            betas=(0.5, 0.999)
        )

        self.disc_optimizer = optim.Adam(
            disc_params,
            lr=float(self.config['disc_lr']),
            betas=(0.5, 0.999)
        )

        # Learning rate schedulers
        self.gen_scheduler = optim.lr_scheduler.StepLR(
            self.gen_optimizer,
            step_size=self.config['lr_decay_epoch'],
            gamma=0.5
        )
        self.disc_scheduler = optim.lr_scheduler.StepLR(
            self.disc_optimizer,
            step_size=self.config['lr_decay_epoch'],
            gamma=0.5
        )

    def setup_loss_functions(self):
        """Setup loss functions"""
        self.recon_criterion = nn.L1Loss()  # Reconstruction loss
        self.adv_criterion = nn.MSELoss()   # Adversarial loss

    def train_epoch(self, epoch):
        """Train for one epoch on single person data"""
        self.model.encoder.train()
        self.model.decoder.train()
        self.model.discriminator.train()

        total_losses = []
        recon_losses = []
        disc_losses = []
        gen_losses = []


        # for batch_idx, batch in enumerate(progress_bar):
        while True:
            response = requests.get(f"{self.config['collector_service_url']}/next-batch")
            if response.json().get("ended"):
              break
            shape = response.json().get("shape", [])
            data = response.json().get("images", [])
            if not data:
              raise ValueError("No data in tensor")
            np_arr = np.array(data, dtype=np.float32)
            batch = torch.from_numpy(np_arr).to(torch.float32)
            if shape:
              batch = batch.reshape(shape)
            batch_idx =response.json().get("batch_idx")
            
            # Get batch data
            real_images = batch.to(self.device)
            batch_size = real_images.size(0)

            # Create labels for adversarial loss
            real_labels = torch.ones(batch_size, 1, 8, 8).to(self.device)
            fake_labels = torch.zeros(batch_size, 1, 8, 8).to(self.device)

            # ========== Train Discriminator ==========
            self.disc_optimizer.zero_grad()

            # Encode and reconstruct
            with torch.no_grad():
                encoded = self.model.encoder(real_images)
                fake_images = self.model.decoder(encoded)

            # Discriminator loss
            disc_real = self.model.discriminator(real_images)
            disc_fake = self.model.discriminator(fake_images.detach())

            disc_loss_real = self.adv_criterion(disc_real, real_labels)
            disc_loss_fake = self.adv_criterion(disc_fake, fake_labels)
            disc_loss = (disc_loss_real + disc_loss_fake) * 0.5

            disc_loss.backward()
            self.disc_optimizer.step()

            # ========== Train Generator ==========
            if batch_idx % self.config['gen_update_freq'] == 0:
                self.gen_optimizer.zero_grad()

                # Encode and reconstruct
                encoded = self.model.encoder(real_images)
                reconstructed = self.model.decoder(encoded)

                # Reconstruction loss
                recon_loss = self.recon_criterion(reconstructed, real_images)

                # Adversarial loss (generator tries to fool discriminator)
                disc_fake = self.model.discriminator(reconstructed)
                gen_adv_loss = self.adv_criterion(disc_fake, real_labels)

                # Total generator loss
                gen_loss = recon_loss * self.config['lambda_recon'] + \
                          gen_adv_loss * self.config['lambda_adv']

                gen_loss.backward()
                self.gen_optimizer.step()

                # Update losses
                total_loss = disc_loss.item() + gen_loss.item()
                total_losses.append(total_loss)
                recon_losses.append(recon_loss.item())
                gen_losses.append(gen_adv_loss.item())

                # Update progress bar
                print({
                    'Total': f'{total_loss:.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'Disc': f'{disc_loss.item():.4f}',
                    'Gen': f'{gen_adv_loss.item():.4f}'
                })

            disc_losses.append(disc_loss.item())

        # Update learning rates
        self.gen_scheduler.step()
        self.disc_scheduler.step()

        # Return average losses
        return {
            'total_loss': np.mean(total_losses) if total_losses else 0,
            'recon_loss': np.mean(recon_losses) if recon_losses else 0,
            'disc_loss': np.mean(disc_losses),
            'gen_loss': np.mean(gen_losses) if gen_losses else 0
        }

    def train(self, person_name, num_epochs=None):
        """Main training loop for single person"""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']

        print(f"Number of epochs: {num_epochs}")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train for one epoch
            losses = self.train_epoch(epoch)

            # Update history
            for key, value in losses.items():
                self.history[key].append(value)

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Total: {losses['total_loss']:.4f} "
                  f"Recon: {losses['recon_loss']:.4f} "
                  f"Disc: {losses['disc_loss']:.4f} "
                  f"Time: {epoch_time:.2f}s", flush=True)

            # Save checkpoint periodically
            if epoch % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(person_name, epoch)

        # Save final models
        self.save_models(person_name)

        return self.history

    def save_checkpoint(self, person_name, epoch):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }

        torch.save(
            checkpoint,
            checkpoint_dir / f'checkpoint_{person_name}_epoch_{epoch:04d}.pth'
        )

    def save_models(self, person_name):
        """Save final models"""
        output_dir = Path(self.config['output_dir'])
        self.model.save_models(output_dir, person_name)
        print(f"Models saved for person: {person_name}")

    def load_checkpoint(self, checkpoint_path, person_name):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])

        self.history = checkpoint['history']

        print(f"Loaded checkpoint for {person_name} from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
