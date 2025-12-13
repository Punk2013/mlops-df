import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import io

from PIL import Image

from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="Storage service")

def discover_existing_persons() -> Dict[str, Path]:
  global person_registry

  for item in Path(config["data_root"]).iterdir():
    if item.is_dir():
      person_name = item.name
      person_registry[person_name] = {
        "path": item,
        "image_count": count_images_in_folder(item),
        "created_at": item.stat().st_ctime,
      }

  print(f"Discovered {len(person_registry)} existing persons:")
  for person, info in person_registry.items():
    print(f"  - {person}: {info['image_count']} images")

  return person_registry

def count_images_in_folder(folder_path: Path) -> int:
  count = 0
  for ext in config["extensions"]:
    count += len(list(folder_path.glob(f"*.{ext}")))
    count += len(list(folder_path.glob(f"*.{ext.upper()}")))
  return count

def load_metadata():
  if Path(config["metadata_file"]).exists():
    try:
      with open(config["metadata_file"], "r") as f:
        metadata = json.load(f)
        for person, info in person_registry.items():
          if person in metadata:
            info.update(metadata[person])
    except Exception as e:
      print(f"Warning: Could not load metadata: {e}")

def save_metadata():
  try:
    with open(config["metadata_file"], "w") as f:
      json.dump(person_registry, f, indent=2, default=str)
  except Exception as e:
    print(f"Warning: Could not save metadata: {e}")

@app.get("/persons", response_model=None)
def list_persons() -> Dict[str, List[str]]:
  return {"persons": list(person_registry.keys())}

def get_person_info(person_name: str) -> Optional[Dict]:
    return person_registry.get(person_name)

def person_exists(person_name: str) -> bool:
  return person_name in person_registry

@app.post("/person", response_model=None)
def create_person_folder(
  person_name: str,
  files: List[UploadFile] = File(),
  image_quality: int = 95,
  target_size: Optional[Tuple[int, int]] = None,
  overwrite: bool = False,
) -> Dict["str", bool]:
  # return false if folder already exists
  global person_registry
  try:
    pil_images: List[Image.Image] = []
    if files:
      for image_bytes in files:
        pil_image = Image.open(image_bytes.file)
        pil_images.append(pil_image)

    person_path = Path(config["data_root"]) / person_name

    if person_path.exists():
      if not overwrite:
        print(
          f"Person '{person_name}' already exists. Use overwrite=True to replace."
        )
        return {"res": False}
      else:
        shutil.rmtree(person_path)
        print(f"Removed existing folder for '{person_name}'")

    person_path.mkdir(parents=True, exist_ok=True)
    print(f"Created folder for '{person_name}' at {person_path}")

    person_registry[person_name] = {
      "path": person_path,
      "image_count": 0,
      "created_at": person_path.stat().st_ctime,
      "image_quality": image_quality,
      "target_size": target_size,
    }

    if pil_images:
      success_count = add_images_to_person(
        person_name, pil_images, image_quality, target_size
      )
      print(f"Added {success_count} images to '{person_name}'")

    save_metadata()

    return {"res": True}

  except Exception as e:
    print(f"Error creating folder for '{person_name}': {e}")
    return {"res": False}

def add_images_to_person(
  person_name: str,
  images: List[Image.Image],
  image_quality: int = 95,
  target_size: tuple = None,
) -> int:
  global person_registry

  if person_name not in person_registry:
    print(f"Person '{person_name}' does not exist. Create folder first.")
    return 0

  person_path = person_registry[person_name]["path"]
  success_count = 0

  for i, img in enumerate(images):
    try:
      if img.mode != "RGB":
        img = img.convert("RGB")

      if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)

      existing_count = person_registry[person_name]["image_count"]
      filename = f"{person_name}_{existing_count + i + 1:06d}.jpg"
      filepath = person_path / filename

      img.save(filepath, "JPEG", quality=image_quality, optimize=True)
      success_count += 1

    except Exception as e:
      print(f"Error saving image {i} for '{person_name}': {e}")
      continue

  person_registry[person_name]["image_count"] += success_count
  person_registry[person_name]["last_updated"] = os.path.getctime(
      person_path
  )
  save_metadata()

  return success_count

def get_person_images(person_name: str) -> List[Path]:
  if person_name not in person_registry:
    return []

  person_path = person_registry[person_name]["path"]
  image_paths = []

  for ext in config["allowed_extensions"]:
    image_paths.extend(person_path.glob(f"*.{ext}"))
    image_paths.extend(person_path.glob(f"*.{ext.upper()}"))

  return sorted(image_paths)

@app.get("/image-count", response_model=None)
def get_image_count(person_name: str) -> Dict[str, int]:
  return {"image_count": person_registry.get(person_name, {}).get("image_count", 0)}

def delete_person(person_name: str) -> bool:
  global person_registry

  if person_name not in person_registry:
    print(f"Person '{person_name}' does not exist.")
    return False

  try:
    person_path = person_registry[person_name]["path"]

    shutil.rmtree(person_path)

    del person_registry[person_name]

    save_metadata()

    print(f"Deleted person '{person_name}' and all images")
    return True

  except Exception as e:
    print(f"Error deleting person '{person_name}': {e}")
    return False

def rename_person(old_name: str, new_name: str) -> bool:
  global person_registry

  if old_name not in person_registry:
    print(f"Person '{old_name}' does not exist.")
    return False

  if new_name in person_registry:
    print(f"Person '{new_name}' already exists.")
    return False

  try:
    old_path = person_registry[old_name]["path"]
    new_path = Path(config["data_root"]) / new_name

    old_path.rename(new_path)

    person_registry[new_name] = person_registry[old_name]
    person_registry[new_name]["path"] = new_path
    del person_registry[old_name]

    rename_image_files(new_name, new_path)

    save_metadata()

    print(f"Renamed '{old_name}' to '{new_name}'")
    return True

  except Exception as e:
    print(f"Error renaming person '{old_name}' to '{new_name}': {e}")
    return False

def rename_image_files(person_name: str, person_path: Path):
  image_paths = get_person_images(person_name)

  for i, old_path in enumerate(image_paths):
    new_filename = f"{person_name}_{i + 1:06d}.jpg"
    new_path = person_path / new_filename
    old_path.rename(new_path)

def create_train_val_split(
  person_names: List[str] = None,
  val_ratio: float = 0.2,
  output_dir: str = None,
) -> bool:
  try:
    if person_names is None:
      person_names = list_persons()

    if output_dir is None:
      output_dir = Path(config["data_root"]) / "train_val_split"
    else:
      output_dir = Path(output_dir)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for person in person_names:
      if person not in person_registry:
        print(f"Warning: Person '{person}' not found, skipping")
        continue

      train_person_dir = train_dir / person
      val_person_dir = val_dir / person
      train_person_dir.mkdir(exist_ok=True)
      val_person_dir.mkdir(exist_ok=True)

      image_paths = get_person_images(person)

      if not image_paths:
        print(f"Warning: No images found for '{person}', skipping")
        continue

      split_idx = int(len(image_paths) * (1 - val_ratio))
      train_images = image_paths[:split_idx]
      val_images = image_paths[split_idx:]

      for img_path in train_images:
        shutil.copy2(img_path, train_person_dir / img_path.name)

      for img_path in val_images:
        shutil.copy2(img_path, val_person_dir / img_path.name)

      print(
          f"Split '{person}': {len(train_images)} train, {len(val_images)} val images"
      )

    print(f"Train/val split created at {output_dir}")
    return True

  except Exception as e:
    print(f"Error creating train/val split: {e}")
    return False

def get_stats() -> Dict:
  stats = {
    "total_persons": len(person_registry),
    "total_images": 0,
    "persons": {},
  }

  for person, info in person_registry.items():
    stats["persons"][person] = info["image_count"]
    stats["total_images"] += info["image_count"]
  return stats

### Initialization
person_registry: dict = {}

with open("config.json", 'r') as config_file:
  config = json.load(config_file)

Path(config["data_root"]).mkdir(parents=True, exist_ok=True)
discover_existing_persons()
load_metadata()
###
