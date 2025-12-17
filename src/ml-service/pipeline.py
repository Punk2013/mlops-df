# training_pipeline.py
import json
from pathlib import Path
import os
import re
import torch
from model import SinglePersonDeepFake
from train import SinglePersonTrainer
import requests

class IncrementalTrainingPipeline:
    def __init__(self, config_path="config.json"):
        """Initialize training pipeline with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_config = self.config['training']

        # Create directories
        Path(self.base_config['output_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.base_config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

        print("Incremental Training Pipeline initialized")
        print(f"Base encoder: {self.base_config.get('encoder_path', 'New encoder')}")

    def train_person(self, person_name, num_epochs=None):
        """
        Train on a single person

        Args:
            person_name: Name of the person (used for saving models)
            person_dir: Directory containing person's images
            num_epochs: Number of epochs (overrides config if provided)
        """
        print(f"\n{'='*60}")
        print(f"Training Person: {person_name}")
        print(f"{'='*60}")

        # Create dataloader
        print("Creating dataloader...")
        params = {
          "name": person_name,
          "batch_size": self.base_config["batch_size"],
          "img_size": self.base_config["img_size"],
          "num_workers": self.base_config["num_workers"]
        }
        response = requests.post(f"{self.base_config['collector_service_url']}/dataloader", params=params)
        assert response.json()["OK"]

        print(f"Batch size: {self.base_config['batch_size']}")

        # Create trainer
        trainer_config = self.base_config.copy()
        trainer_config['person_name'] = person_name

        # Set encoder path for this training session
        encoder_path = self._get_latest_encoder_path()
        if encoder_path:
            trainer_config['encoder_path'] = str(encoder_path)

        trainer = SinglePersonTrainer(trainer_config)

        # Train
        if num_epochs is None:
            num_epochs = self.base_config['num_epochs']

        history = trainer.train(person_name, num_epochs)

        print(f"\nTraining completed for {person_name}!")

        return history

    def _get_latest_encoder_path(self):
        """Get path to the most recent encoder"""
        output_dir = Path(self.base_config['output_dir'])

        # Look for encoder files
        encoder_files = list(output_dir.glob("encoder_*.pth"))

        if not encoder_files:
            return None

        # Get the most recent encoder by modification time
        latest_encoder = max(encoder_files, key=lambda x: x.stat().st_mtime)
        return latest_encoder

    def train_multiple_persons(self, persons_dict):
        """
        Train on multiple persons sequentially

        Args:
            persons_dict: Dictionary of {person_name: person_dir}
        """
        all_histories = {}

        for i, (person_name, person_dir) in enumerate(persons_dict.items()):
            print(f"\nTraining person {i+1}/{len(persons_dict)}: {person_name}")

            # Train this person
            history = self.train_person(person_name, person_dir)
            all_histories[person_name] = history

            # Print summary
            final_loss = history['total_loss'][-1] if history['total_loss'] else 0
            print(f"Final loss for {person_name}: {final_loss:.4f}")

        print(f"\nAll {len(persons_dict)} persons trained successfully!")
        return all_histories

    def load_person_models(self, person_name, encoder_path=None):
        """Load models for a specific person"""
        model_dir = Path(self.base_config['output_dir'])

        model = SinglePersonDeepFake(
            encoder_path=encoder_path if encoder_path else model_dir / f"encoder_{person_name}.pth",
            device=self.device
        )

        # Load decoder and discriminator
        decoder_path = model_dir / f"decoder_{person_name}.pth"
        disc_path = model_dir / f"discriminator_{person_name}.pth"

        if decoder_path.exists():
            checkpoint = torch.load(decoder_path, map_location=self.device)
            model.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if disc_path.exists():
            checkpoint = torch.load(disc_path, map_location=self.device)
            model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        return model
    
    def list_trained_persons(self):
      models_dir = self.base_config['output_dir']
      names = []
      for filename in os.listdir(models_dir):
        match = re.match(r'decoder_(.*)\.pth$', filename)
        if match:
          names.append(match.group(1))
      
      return names
      
    def remove_person_model(self, person_name: str):
      os.remove(Path(self.base_config['output_dir']) / Path(f"encoder_{person_name}.pth"))
      os.remove(Path(self.base_config['output_dir']) / Path(f"decoder_{person_name}.pth"))
      os.remove(Path(self.base_config['output_dir']) / Path(f"discriminator_{person_name}.pth"))