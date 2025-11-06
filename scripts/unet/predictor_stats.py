import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import pandas as pd
from traffic_ai.unet.dataset_loader import LaneDataset
from traffic_ai.unet.model import UNet
from pathlib import Path

# --- Paths and Settings ---
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")
IMAGE_DIR = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/images")
MASK_DIR = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/masks")
MODEL_PATH = str(ROOT / "models/unet/unet_checkpoint.pth")
IMAGE_SIZE = (256, 512)
BATCH_SIZE = 4

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform ---
transform = A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# --- Dataset and Dataloader ---
dataset = LaneDataset(IMAGE_DIR, MASK_DIR, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Load Model ---
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Metrics ---
def iou_score(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    intersection = (pred & target).float().sum((1, 2, 3))
    union = (pred | target).float().sum((1, 2, 3))
    return (intersection / (union + 1e-6)).mean().item()

def dice_score(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    intersection = (pred * target).float().sum((1, 2, 3))
    return (2. * intersection / (pred.float().sum((1, 2, 3)) + target.float().sum((1, 2, 3)) + 1e-6)).mean().item()

def pixel_accuracy(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    correct = (pred == target).float().sum((1, 2, 3))
    total = torch.ones_like(pred).float().sum((1, 2, 3))
    return (correct / total).mean().item()

# --- Evaluation ---
results = []
ious, dices, accs = [], [], []

with torch.no_grad():
    for idx, (images, masks) in enumerate(tqdm(data_loader)):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs)

        batch_iou = iou_score(outputs, masks)
        batch_dice = dice_score(outputs, masks)
        batch_acc = pixel_accuracy(outputs, masks)

        ious.append(batch_iou)
        dices.append(batch_dice)
        accs.append(batch_acc)

        results.append({
            "Batch": f"Batch_{idx+1}",
            "IoU": batch_iou,
            "Dice Coefficient": batch_dice,
            "Pixel Accuracy": batch_acc
        })

# --- Save CSV ---
df = pd.DataFrame(results)
out_csv = ROOT / "outputs/evaluation_metrics.csv"
df.to_csv(out_csv, index=False)
print(f"Evaluation results saved to {out_csv}")

# --- Plotting ---
plt.figure(figsize=(16, 6))

# IoU
plt.subplot(1, 3, 1)
plt.plot(df["Batch"], df["IoU"], marker='o', label="IoU")
plt.axhline(df["IoU"].mean(), color='r', linestyle='--', label=f"Mean IoU: {df['IoU'].mean():.2f}")
plt.xticks(rotation=90, fontsize=6)
plt.title("IoU per Batch")
plt.xlabel("Batch")
plt.ylabel("IoU")
plt.legend()
plt.grid(True)

# Pixel Accuracy
plt.subplot(1, 3, 2)
plt.plot(df["Batch"], df["Pixel Accuracy"], marker='s', color='g', label="Pixel Accuracy")
plt.axhline(df["Pixel Accuracy"].mean(), color='r', linestyle='--', label=f"Mean Accuracy: {df['Pixel Accuracy'].mean():.2f}")
plt.xticks(rotation=90, fontsize=6)
plt.title("Pixel Accuracy per Batch")
plt.xlabel("Batch")
plt.ylabel("Pixel Accuracy")
plt.legend()
plt.grid(True)

# Dice
plt.subplot(1, 3, 3)
plt.plot(df["Batch"], df["Dice Coefficient"], marker='^', color='orange', label="Dice Coefficient")
plt.axhline(df["Dice Coefficient"].mean(), color='r', linestyle='--', label=f"Mean Dice: {df['Dice Coefficient'].mean():.2f}")
plt.xticks(rotation=90, fontsize=6)
plt.title("Dice Coefficient per Batch")
plt.xlabel("Batch")
plt.ylabel("Dice Coefficient")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
