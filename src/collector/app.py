# single_person_dataset.py
import os
import json
from PIL import Image
from starlette.responses import Response
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import requests
from tqdm import tqdm

from fastapi import FastAPI, File, UploadFile
import zipfile
import io

app = FastAPI(title="Collector service")

config = {
  "storage_service_url": "http://storage-service:8001"
}

dataloader = None
data_iter = None
progress_bar = None

@app.post("/dataloader", response_model=None)
def post_dataloader(
  name: str,
  batch_size: int,
  img_size: int,
  num_workers: int,
  augment=True
):
  global dataloader
  try:
    
    person_dir = f"data/{name}"
    response = requests.get(f"{config['storage_service_url']}/person-folder/{name}")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
      zf.extractall(person_dir)
    
    dataloader = create_dataloader(
        person_dir,
        batch_size,
        img_size,
        num_workers,
        augment
    )
    progress_bar = tqdm(dataloader)
    data_iter = enumerate(progress_bar)
    return {"OK": True}
  except Exception as e:
    return {"OK": (False, e)}

@app.get("/next-batch")
def get_batch():
  try:
    batch_idx, batch = next(data_iter)
    batch_dict = {
      "batch_idx": batch_idx,
      "images": batch["image"].tolist(),
      "shape": batch["image"].shape,
      "paths": batch["path"].tolist(),
      "ended": False
    }
    return batch_dict
  except StopIteration:
    return {"ended": True}
  

class SinglePersonDataset(Dataset):
    def __init__(self, person_dir, transform=None, img_size=128, augment=True):
        """
        Dataset for single person training

        Args:
            person_dir: Directory containing person's images
            transform: Custom transforms
            img_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.person_dir = person_dir
        self.img_size = img_size
        self.augment = augment

        # Get all image paths
        self.image_paths = self._get_image_paths(person_dir)
        assert len(self.image_paths) > 0, f"No images found in {person_dir}"
        print(f"Found {len(self.image_paths)} images in {person_dir}")

        # Setup transforms
        self.transform = transform if transform else self._get_default_transforms()

    def _get_image_paths(self, directory):
        """Get all image file paths from directory"""
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
            image_paths.extend(glob.glob(os.path.join(directory, f'*.{ext.upper()}')))
        return sorted(image_paths)

    def _get_default_transforms(self):
        """Get default transforms with augmentation"""
        if self.augment:
            return transforms.Compose([
                transforms.Resize((self.img_size + 20, self.img_size + 20)),
                transforms.RandomCrop((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Returns tensor image and its path"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return {'image': img, 'path': img_path}

def create_dataloader(person_dir, batch_size=8, img_size=128,
                      num_workers=4, augment=True):
    """
    Create dataloader for single person

    Args:
        person_dir: Directory containing person's images
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of workers
        augment: Whether to apply augmentation
    """
    dataset = SinglePersonDataset(
        person_dir=person_dir,
        img_size=img_size,
        augment=augment
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader
