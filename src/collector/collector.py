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
        self.data_root = data_root
        self.img_size = img_size
        self.augment = augment

        self.person_a_paths = self._get_image_paths(
            os.path.join(data_root, person_a_dir)
        )
        self.person_b_paths = self._get_image_paths(
            os.path.join(data_root, person_b_dir)
        )

        assert len(self.person_a_paths) > 0, f"No images found in {person_a_dir}"
        assert len(self.person_b_paths) > 0, f"No images found in {person_b_dir}"

        print(f"Found {len(self.person_a_paths)} images for person A")
        print(f"Found {len(self.person_b_paths)} images for person B")

        self.transform = transform if transform else self._get_default_transforms()

    def _get_image_paths(self, directory):
        extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
            image_paths.extend(glob.glob(os.path.join(directory, f"*.{ext.upper()}")))
        return sorted(image_paths)

    def _get_default_transforms(self):
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
        return max(len(self.person_a_paths), len(self.person_b_paths))

    def __getitem__(self, idx):
        a_idx = idx % len(self.person_a_paths)
        b_idx = idx % len(self.person_b_paths)

        person_a_img = Image.open(self.person_a_paths[a_idx]).convert("RGB")
        person_b_img = Image.open(self.person_b_paths[b_idx]).convert("RGB")

        if self.transform:
            person_a_img = self.transform(person_a_img)
            person_b_img = self.transform(person_b_img)

        return {
            "person_a": person_a_img,
            "person_b": person_b_img,
            "a_path": self.person_a_paths[a_idx],
            "b_path": self.person_b_paths[b_idx],
        }


class FaceDataLoader:
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
        train_root = os.path.join(data_root, "train")
        val_root = os.path.join(data_root, "val")

        if not os.path.exists(train_root):
            print("No train/val split found, using entire dataset with random split")
            return FaceDataLoader._create_random_split_dataloader(
                data_root,
                batch_size,
                img_size,
                num_workers,
                augment,
            )

        train_dataset = FaceSwapDataset(train_root, img_size=img_size, augment=augment)
        val_dataset = FaceSwapDataset(
            val_root,
            img_size=img_size,
            augment=False,
        )

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
        data_root, batch_size, img_size, num_workers, augment
    ):
        full_dataset = FaceSwapDataset(data_root, img_size=img_size, augment=augment)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

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


def test_dataloader():
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

    train_loader, val_loader = FaceDataLoader.create_dataloaders(
        data_root=data_root, batch_size=8, img_size=128, num_workers=2, augment=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    for batch in train_loader:
        person_a = batch["person_a"]
        person_b = batch["person_b"]

        print(f"Batch - Person A shape: {person_a.shape}")
        print(f"Batch - Person B shape: {person_b.shape}")
        print(f"Person A range: [{person_a.min():.3f}, {person_a.max():.3f}]")
        print(f"Person B range: [{person_b.min():.3f}, {person_b.max():.3f}]")

        break


if __name__ == "__main__":
    test_dataloader()
