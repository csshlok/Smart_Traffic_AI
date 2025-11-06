import os
import cv2
import numpy as np
import torch
from traffic_ai.unet.model import UNet
from traffic_ai.unet.dataset_loader import LaneDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from traffic_ai.unet.attention_model import Attention_UNet
from pathlib import Path

# === Config ===
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")
IMAGE_DIR = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/images")
MASK_DIR = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/masks")
MODEL_PATH = str(ROOT / "models/unet/attention_unet_checkpoint.pth")
IMAGE_SIZE = (256, 512)
BATCH_SIZE = 1
THRESHOLDS = [0.3, 0.5, 0.7]
OUTPUT_DIR = str(ROOT / "outputs/threshold_overlay")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform = A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# === Dataset & Loader ===
dataset = LaneDataset(IMAGE_DIR, MASK_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Model ===
model = Attention_UNet(in_channels=4, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Multi-Threshold Overlay Function ===
def overlay_thresholds(original, pred_mask, thresholds):
    colors = [(0, 255, 255), (0, 165, 255), (0, 0, 255)]  # yellow, orange, red
    overlay = original.copy()
    for i, t in enumerate(thresholds):
        binary = (pred_mask > t).astype(np.uint8) * 255
        colored = np.zeros_like(original)
        for c in range(3):
            colored[:, :, c] = binary * (colors[i][c] / 255.0)
        overlay = cv2.addWeighted(overlay, 1.0, colored.astype(np.uint8), 0.3, 0)
    return overlay

# === Inference & Save ===
with torch.no_grad():
    for idx, (image, _) in enumerate(loader):
        image_tensor = image.to(device)
        original = (image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

        overlaid_img = overlay_thresholds(original, pred_mask, THRESHOLDS)
        save_path = os.path.join(OUTPUT_DIR, f"overlay_{idx:04}.png")
        cv2.imwrite(save_path, overlaid_img)

print(f"[✓] Overlay images saved in: {OUTPUT_DIR}")
