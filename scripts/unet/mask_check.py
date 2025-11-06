import os
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# Check a specific mask first
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")
mask = cv2.imread(str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/masks/19.png"), cv2.IMREAD_GRAYSCALE)
print(f"Max: {mask.max()}, Min: {mask.min()}, Mean: {mask.mean()}, Unique: {np.unique(mask)}")

# Define directories
image_dir = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/images")
mask_dir = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/masks")

image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

# Extract stems (no extensions)
image_stems = {os.path.splitext(f)[0]: f for f in image_files}
mask_stems = {os.path.splitext(f)[0]: f for f in mask_files}

print(f"Images: {len(image_files)}")
print(f"Masks: {len(mask_files)}\n")

valid_pairs = 0
blank_masks = 0

# Match using stems
for stem, img_filename in image_stems.items():
    if stem in mask_stems:
        mask_path = os.path.join(mask_dir, mask_stems[stem])
        mask = np.array(Image.open(mask_path).convert("L"))
        if mask.max() == 0:
            blank_masks += 1
        else:
            valid_pairs += 1

print(f"→ Total valid image-mask pairs: {valid_pairs}")
print(f"→ Total blank masks (no lane): {blank_masks}")
print(f"→ Missing masks for images: {len(image_stems) - len(mask_stems)}")
