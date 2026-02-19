import os
import shutil
import random
from pathlib import Path

# Paths
raw_dir = Path("data/datasets/raw")
processed_dir = Path("data/datasets/processed")
train_dir = processed_dir / "train"
val_dir = processed_dir / "val"

# Split ratio (80% train, 20% validation)
split_ratio = 0.8

# Create processed directories
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Loop through each class folder in raw
for class_folder in raw_dir.iterdir():
    if not class_folder.is_dir():
        continue   # skip files, only process folders

    class_name = class_folder.name
    images = list(class_folder.glob("*.*"))  # get all files in class folder
    random.shuffle(images)                    # shuffle for randomness

    # Calculate split index
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Create class subfolders in train and val
    (train_dir / class_name).mkdir(parents=True, exist_ok=True)
    (val_dir / class_name).mkdir(parents=True, exist_ok=True)

    # Copy images to train folder
    for img in train_images:
        shutil.copy(img, train_dir / class_name / img.name)

    # Copy images to val folder
    for img in val_images:
        shutil.copy(img, val_dir / class_name / img.name)

    print(f"Class '{class_name}': {len(train_images)} train, {len(val_images)} val")

print("Done! Data split into train/val.")