import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from traffic_ai.unet.dataset_loader import LaneDataset
from traffic_ai.unet.model_se import UNet_SE
import os
from torch.utils.data import random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from traffic_ai.unet.attention_model import Attention_UNet
from traffic_ai.unet.model import UNet
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")
IMAGE_DIR = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/images")
MASK_DIR = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/masks")
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
IMAGE_SIZE = (256, 256)
CHECKPOINT_PATH = str(ROOT / "models/unet/attention4_unet_checkpoint.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

def enhance_contrast_if_dark(image_np, **kawargs):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    if gray.mean() < 80:  # low brightness threshold
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        lab_eq = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return enhanced
    return image_np

def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    TP = (pred_flat * target_flat).sum()
    FP = ((1 - target_flat) * pred_flat).sum()
    FN = (target_flat * (1 - pred_flat)).sum()
    return 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

# Transformations
train_transform = A.Compose([
    A.Lambda(image=enhance_contrast_if_dark),  # conditional contrast
    A.Resize(*IMAGE_SIZE),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Lambda(image=enhance_contrast_if_dark),
    A.Resize(*IMAGE_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])



# Dataset and DataLoader
dataset = LaneDataset(IMAGE_DIR, MASK_DIR, transform=train_transform)
#data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
val_dataset.dataset.transform = val_transform
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = UNet(in_channels=3).to(device)

bce_loss = nn.BCEWithLogitsLoss()
#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)



# Loss and Optimizer
#criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images = images.to(device)
        masks = masks.to(device)
        #images = torch.stack([enhance_contrast(img) for img in images]).to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = bce_loss(outputs, masks) + tversky_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Backward pass and optimization

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), CHECKPOINT_PATH)

print("Training complete.")
