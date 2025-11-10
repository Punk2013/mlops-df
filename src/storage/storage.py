import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


class PersonDataManager:
    def __init__(self, data_root: str, allowed_extensions: List[str] = None):
        self.data_root = Path(data_root)
        self.allowed_extensions = allowed_extensions or [
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tiff",
        ]

        self.data_root.mkdir(parents=True, exist_ok=True)

        self.person_registry = {}
        self._discover_existing_persons()

        self.metadata_file = self.data_root / "person_metadata.json"
        self._load_metadata()

    def _discover_existing_persons(self) -> Dict[str, Path]:
        self.person_registry = {}

        for item in self.data_root.iterdir():
            if item.is_dir():
                person_name = item.name
                self.person_registry[person_name] = {
                    "path": item,
                    "image_count": self._count_images_in_folder(item),
                    "created_at": item.stat().st_ctime,
                }

        print(f"Discovered {len(self.person_registry)} existing persons:")
        for person, info in self.person_registry.items():
            print(f"  - {person}: {info['image_count']} images")

        return self.person_registry

    def _count_images_in_folder(self, folder_path: Path) -> int:
        count = 0
        for ext in self.allowed_extensions:
            count += len(list(folder_path.glob(f"*.{ext}")))
            count += len(list(folder_path.glob(f"*.{ext.upper()}")))
        return count

    def _load_metadata(self):
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
                for person, info in self.person_registry.items():
                    if person in metadata:
                        info.update(metadata[person])
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")

    def _save_metadata(self):
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.person_registry, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")

    def list_persons(self) -> List[str]:
        return list(self.person_registry.keys())

    def get_person_info(self, person_name: str) -> Optional[Dict]:
        return self.person_registry.get(person_name)

    def person_exists(self, person_name: str) -> bool:
        return person_name in self.person_registry

    def create_person_folder(
        self,
        person_name: str,
        images: List[Image.Image] = None,
        image_quality: int = 95,
        target_size: tuple = None,
        overwrite: bool = False,
    ) -> bool:
        # return false if folder already exists
        try:
            person_path = self.data_root / person_name

            if person_path.exists():
                if not overwrite:
                    print(
                        f"Person '{person_name}' already exists. Use overwrite=True to replace."
                    )
                    return False
                else:
                    shutil.rmtree(person_path)
                    print(f"Removed existing folder for '{person_name}'")

            person_path.mkdir(parents=True, exist_ok=True)
            print(f"Created folder for '{person_name}' at {person_path}")

            self.person_registry[person_name] = {
                "path": person_path,
                "image_count": 0,
                "created_at": person_path.stat().st_ctime,
                "image_quality": image_quality,
                "target_size": target_size,
            }

            if images:
                success_count = self.add_images_to_person(
                    person_name, images, image_quality, target_size
                )
                print(f"Added {success_count} images to '{person_name}'")

            self._save_metadata()

            return True

        except Exception as e:
            print(f"Error creating folder for '{person_name}': {e}")
            return False

    def add_images_to_person(
        self,
        person_name: str,
        images: List[Image.Image],
        image_quality: int = 95,
        target_size: tuple = None,
    ) -> int:
        if person_name not in self.person_registry:
            print(f"Person '{person_name}' does not exist. Create folder first.")
            return 0

        person_path = self.person_registry[person_name]["path"]
        success_count = 0

        for i, img in enumerate(images):
            try:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                if target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)

                existing_count = self.person_registry[person_name]["image_count"]
                filename = f"{person_name}_{existing_count + i + 1:06d}.jpg"
                filepath = person_path / filename

                img.save(filepath, "JPEG", quality=image_quality, optimize=True)
                success_count += 1

            except Exception as e:
                print(f"Error saving image {i} for '{person_name}': {e}")
                continue

        self.person_registry[person_name]["image_count"] += success_count
        self.person_registry[person_name]["last_updated"] = os.path.getctime(
            person_path
        )
        self._save_metadata()

        return success_count

    def get_person_images(self, person_name: str) -> List[Path]:
        if person_name not in self.person_registry:
            return []

        person_path = self.person_registry[person_name]["path"]
        image_paths = []

        for ext in self.allowed_extensions:
            image_paths.extend(person_path.glob(f"*.{ext}"))
            image_paths.extend(person_path.glob(f"*.{ext.upper()}"))

        return sorted(image_paths)

    def get_image_count(self, person_name: str) -> int:
        return self.person_registry.get(person_name, {}).get("image_count", 0)

    def delete_person(self, person_name: str) -> bool:
        if person_name not in self.person_registry:
            print(f"Person '{person_name}' does not exist.")
            return False

        try:
            person_path = self.person_registry[person_name]["path"]

            shutil.rmtree(person_path)

            del self.person_registry[person_name]

            self._save_metadata()

            print(f"Deleted person '{person_name}' and all images")
            return True

        except Exception as e:
            print(f"Error deleting person '{person_name}': {e}")
            return False

    def rename_person(self, old_name: str, new_name: str) -> bool:
        if old_name not in self.person_registry:
            print(f"Person '{old_name}' does not exist.")
            return False

        if new_name in self.person_registry:
            print(f"Person '{new_name}' already exists.")
            return False

        try:
            old_path = self.person_registry[old_name]["path"]
            new_path = self.data_root / new_name

            old_path.rename(new_path)

            self.person_registry[new_name] = self.person_registry[old_name]
            self.person_registry[new_name]["path"] = new_path
            del self.person_registry[old_name]

            self._rename_image_files(new_name, new_path)

            self._save_metadata()

            print(f"Renamed '{old_name}' to '{new_name}'")
            return True

        except Exception as e:
            print(f"Error renaming person '{old_name}' to '{new_name}': {e}")
            return False

    def _rename_image_files(self, person_name: str, person_path: Path):
        image_paths = self.get_person_images(person_name)

        for i, old_path in enumerate(image_paths):
            new_filename = f"{person_name}_{i + 1:06d}.jpg"
            new_path = person_path / new_filename
            old_path.rename(new_path)

    def create_train_val_split(
        self,
        person_names: List[str] = None,
        val_ratio: float = 0.2,
        output_dir: str = None,
    ) -> bool:
        try:
            if person_names is None:
                person_names = self.list_persons()

            if output_dir is None:
                output_dir = self.data_root / "train_val_split"
            else:
                output_dir = Path(output_dir)

            train_dir = output_dir / "train"
            val_dir = output_dir / "val"
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)

            for person in person_names:
                if person not in self.person_registry:
                    print(f"Warning: Person '{person}' not found, skipping")
                    continue

                train_person_dir = train_dir / person
                val_person_dir = val_dir / person
                train_person_dir.mkdir(exist_ok=True)
                val_person_dir.mkdir(exist_ok=True)

                image_paths = self.get_person_images(person)

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

    def get_stats(self) -> Dict:
        stats = {
            "total_persons": len(self.person_registry),
            "total_images": 0,
            "persons": {},
        }

        for person, info in self.person_registry.items():
            stats["persons"][person] = info["image_count"]
            stats["total_images"] += info["image_count"]

        return stats


def test():
    data_manager = PersonDataManager("./deepfake_data")

    sample_images = []
    for i in range(5):
        img = Image.new("RGB", (128, 128), color=(i * 50, i * 30, i * 70))
        sample_images.append(img)

    success = data_manager.create_person_folder(
        person_name="john_doe",
        images=sample_images,
        image_quality=90,
        target_size=(256, 256),
    )

    if success:
        print("Successfully created person folder!")

    persons = data_manager.list_persons()
    print(f"Available persons: {persons}")

    info = data_manager.get_person_info("john_doe")
    print(f"John Doe info: {info}")

    more_images = [
        Image.new("RGB", (128, 128), color=(100, 150, 200)) for _ in range(3)
    ]
    added_count = data_manager.add_images_to_person("john_doe", more_images)
    print(f"Added {added_count} more images")

    stats = data_manager.get_stats()
    print(f"Data stats: {stats}")

    data_manager.create_train_val_split(val_ratio=0.2)


if __name__ == "__main__":
    test()
