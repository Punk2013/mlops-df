import glob
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class FaceSwapDataset(Dataset):
    def __init__(
        self,
        data_root,
        person_a_dir="person_A",
        person_b_dir="person_B",
        transform=None,
        img_size=128,
        augment=True,
    ):
        """
        Dataset for deepfake face swapping

        Args:
            data_root: Root directory containing person_A and person_B folders
            person_a_dir: Directory for person A images
            person_b_dir: Directory for person B images
            transform: Custom transforms (if None, default will be used)
            img_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.data_root = data_root
        self.img_size = img_size
        self.augment = augment

        # Get image paths for both persons
        self.person_a_paths = self._get_image_paths(
            os.path.join(data_root, person_a_dir)
        )
        self.person_b_paths = self._get_image_paths(
            os.path.join(data_root, person_b_dir)
        )

        # Make sure we have images
        assert len(self.person_a_paths) > 0, f"No images found in {person_a_dir}"
        assert len(self.person_b_paths) > 0, f"No images found in {person_b_dir}"

        print(f"Found {len(self.person_a_paths)} images for person A")
        print(f"Found {len(self.person_b_paths)} images for person B")

        # Setup transforms
        self.transform = transform if transform else self._get_default_transforms()

    def _get_image_paths(self, directory):
        """Get all image file paths from directory"""
        extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
            image_paths.extend(glob.glob(os.path.join(directory, f"*.{ext.upper()}")))
        return sorted(image_paths)

    def _get_default_transforms(self):
        """Get default transforms with augmentation"""
        if self.augment:
            return transforms.Compose(
                [
                    transforms.Resize((self.img_size + 20, self.img_size + 20)),
                    transforms.RandomCrop((self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def __len__(self):
        """Return length of dataset (max of both persons)"""
        return max(len(self.person_a_paths), len(self.person_b_paths))

    def __getitem__(self, idx):
        """
        Returns:
            person_a: Tensor of person A image
            person_b: Tensor of person B image
            a_path: Path to person A image
            b_path: Path to person B image
        """
        # Use modulo to handle different dataset sizes
        a_idx = idx % len(self.person_a_paths)
        b_idx = idx % len(self.person_b_paths)

        # Load images
        person_a_img = Image.open(self.person_a_paths[a_idx]).convert("RGB")
        person_b_img = Image.open(self.person_b_paths[b_idx]).convert("RGB")

        # Apply transforms
        if self.transform:
            person_a = self.transform(person_a_img)
            person_b = self.transform(person_b_img)

        return {
            "person_a": person_a,
            "person_b": person_b,
            "a_path": self.person_a_paths[a_idx],
            "b_path": self.person_b_paths[b_idx],
        }


class PairedFaceDataset(Dataset):
    """Dataset for aligned face pairs (for supervised training)"""

    def __init__(self, data_root, pair_file=None, transform=None, img_size=128):
        self.data_root = data_root
        self.img_size = img_size
        self.pairs = []

        # Load pairs from file or create from directory structure
        if pair_file and os.path.exists(pair_file):
            self._load_pairs_from_file(pair_file)
        else:
            self._discover_pairs()

        self.transform = transform if transform else self._get_default_transforms()

    def _load_pairs_from_file(self, pair_file):
        """Load image pairs from a text file"""
        with open(pair_file, "r") as f:
            for line in f:
                a_path, b_path = line.strip().split()
                self.pairs.append((a_path, b_path))

    def _discover_pairs(self):
        """Discover paired images from directory structure"""
        # Assuming structure: data_root/pairs/pair_001/{a.jpg, b.jpg}
        pair_dirs = glob.glob(os.path.join(self.data_root, "pairs", "pair_*"))

        for pair_dir in pair_dirs:
            a_path = os.path.join(pair_dir, "a.jpg")
            b_path = os.path.join(pair_dir, "b.jpg")

            if os.path.exists(a_path) and os.path.exists(b_path):
                self.pairs.append((a_path, b_path))

    def _get_default_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_path, b_path = self.pairs[idx]

        a_img = Image.open(a_path).convert("RGB")
        b_img = Image.open(b_path).convert("RGB")

        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        return {
            "person_a": a_img,
            "person_b": b_img,
            "a_path": a_path,
            "b_path": b_path,
        }


class FaceDataLoader:
    """Wrapper class for creating dataloaders"""

    @staticmethod
    def create_dataloaders(
        data_root,
        batch_size=8,
        img_size=128,
        num_workers=4,
        augment=True,
        paired_data=False,
        pair_file=None,
    ):
        """
        Create train and validation dataloaders

        Args:
            data_root: Root directory containing data
            batch_size: Batch size for training
            img_size: Target image size
            num_workers: Number of workers for data loading
            augment: Whether to apply data augmentation
            paired_data: Whether to use paired dataset
            pair_file: Path to pair file for paired dataset
        """

        # Split data root for train/val if needed
        train_root = os.path.join(data_root, "train")
        val_root = os.path.join(data_root, "val")

        # If no train/val split, create random split
        if not os.path.exists(train_root):
            print("No train/val split found, using entire dataset with random split")
            return FaceDataLoader._create_random_split_dataloader(
                data_root,
                batch_size,
                img_size,
                num_workers,
                augment,
                paired_data,
                pair_file,
            )

        # Create datasets
        if paired_data:
            train_dataset = PairedFaceDataset(train_root, pair_file, img_size=img_size)
            val_dataset = PairedFaceDataset(val_root, pair_file, img_size=img_size)
        else:
            train_dataset = FaceSwapDataset(
                train_root, img_size=img_size, augment=augment
            )
            val_dataset = FaceSwapDataset(
                val_root,
                img_size=img_size,
                augment=False,  # No augmentation for val
            )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader

    @staticmethod
    def _create_random_split_dataloader(
        data_root, batch_size, img_size, num_workers, augment, paired_data, pair_file
    ):
        """Create dataloader with random train/val split"""

        if paired_data:
            full_dataset = PairedFaceDataset(data_root, pair_file, img_size=img_size)
        else:
            full_dataset = FaceSwapDataset(
                data_root, img_size=img_size, augment=augment
            )

        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader


# Example usage and visualization
def test_dataloader():
    """Test the dataloader implementation"""

    # data/
    #   train/
    #     person_A/
    #       image1.jpg, image2.jpg, ...
    #     person_B/
    #       image1.jpg, image2.jpg, ...
    #   val/
    #     person_A/
    #       ...
    #     person_B/
    #       ...

    data_root = "./data"

    # Create dataloaders
    train_loader, val_loader = FaceDataLoader.create_dataloaders(
        data_root=data_root, batch_size=8, img_size=128, num_workers=2, augment=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    for batch in train_loader:
        person_a = batch["person_a"]
        person_b = batch["person_b"]

        print(f"Batch - Person A shape: {person_a.shape}")
        print(f"Batch - Person B shape: {person_b.shape}")
        print(f"Person A range: [{person_a.min():.3f}, {person_a.max():.3f}]")
        print(f"Person B range: [{person_b.min():.3f}, {person_b.max():.3f}]")

        break


def denormalize(tensor):
    """Convert normalized tensor back to image"""
    return tensor * 0.5 + 0.5


if __name__ == "__main__":
    test_dataloader()
