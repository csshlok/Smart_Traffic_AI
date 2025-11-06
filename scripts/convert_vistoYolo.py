import os
import shutil
import cv2
from pathlib import Path

# ========== CONFIG ==========
# Change these to your actual paths
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")
RAW_DATA_ROOT = str(DATA_ROOT / "datasets/visdrone/raw")
YOLO_DATA_ROOT = str(DATA_ROOT / "datasets/visdrone/yolo")

SETS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val"
}

# Map VisDrone categories to YOLO class IDs
VISDRONE_TO_YOLO = {
    1: 0,  # pedestrian
    2: 0,  # person
    3: 1,  # bicycle
    4: 2,  # car
    5: 3,  # van
    6: 4,  # truck
    7: 5,  # tricycle
    8: 6,  # awning-tricycle
    9: 7,  # bus
    10: 8  # motor
}

# ========== SCRIPT ==========

for split, folder in SETS.items():
    img_dir = os.path.join(RAW_DATA_ROOT, folder, "images")
    ann_dir = os.path.join(RAW_DATA_ROOT, folder, "annotations")

    out_img_dir = os.path.join(YOLO_DATA_ROOT, "images", split)
    out_lbl_dir = os.path.join(YOLO_DATA_ROOT, "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for label_file in os.listdir(ann_dir):
        base_name = os.path.splitext(label_file)[0]
        img_path = os.path.join(img_dir, base_name + ".jpg")
        ann_path = os.path.join(ann_dir, label_file)
        out_label_path = os.path.join(out_lbl_dir, base_name + ".txt")

        # Skip missing images
        if not os.path.exists(img_path):
            continue

        # Copy image to YOLO dataset
        shutil.copy(img_path, os.path.join(out_img_dir, base_name + ".jpg"))

        with open(ann_path, "r") as f:
            lines = f.readlines()

        yolo_labels = []
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        for line in lines:
            parts = line.strip().split(',')

            if len(parts) != 8:
                continue

            try:
                vals = list(map(int,parts))
            except ValueError:
                continue

            xmin, ymin, width, height = vals[0], vals[1], vals[2], vals[3]
            class_id = vals[5]

            if class_id not in VISDRONE_TO_YOLO:
                continue  # skip ignored/other

            yolo_cls = VISDRONE_TO_YOLO[class_id]

            # Convert to YOLO format
            x_center = (xmin + width / 2) / w
            y_center = (ymin + height / 2) / h
            norm_width = width / w
            norm_height = height / h

            yolo_labels.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

        # Write new YOLO annotation
        with open(out_label_path, "w") as out_f:
            out_f.write("\n".join(yolo_labels))
