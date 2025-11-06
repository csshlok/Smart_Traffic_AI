import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from traffic_ai.unet.model import UNet
from traffic_ai.unet.dataset_loader import LaneDataset
from torch.utils.data import DataLoader
import random
from traffic_ai.unet.attention_model import Attention_UNet
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")
IMAGE_DIR = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/images")
MASK_DIR = str(DATA_ROOT / "datasets/TuSimple/TUSimple/lane_dataset/masks")
MODEL_PATH = str(ROOT / "models/unet/unet_checkpoint.pth")
IMAGE_SIZE = (256, 256)

# -----------------------------
# Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# Load validation data
# -----------------------------
dataset = LaneDataset(IMAGE_DIR, MASK_DIR, image_size=IMAGE_SIZE)
val_size = int(0.2 * len(dataset))
_, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# -----------------------------
# Inference & Visualization
# -----------------------------
def predict_and_visualize(model, dataloader, device, num_samples=5):
    for i, (image, mask) in enumerate(dataloader):
        if i >= num_samples:
            break
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
            pred_mask = torch.sigmoid(output).cpu().numpy()[0][0]
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        orig_img = image.cpu().numpy()[0].transpose(1, 2, 0) * 255
        orig_img = orig_img.astype(np.uint8)
        mask_img = pred_mask
        cv2.imwrite("output_pred_mask.png", (pred_mask * 255).astype(np.uint8))
        overlay = orig_img.copy()
        overlay[pred_mask.squeeze() > 0.5] = [255, 0, 0]  # red overlay



        # Display
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(orig_img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Predicted Lane Mask")
        plt.imshow(mask_img, cmap='gray')
        plt.axis("off")

        plt.tight_layout()
        plt.show()
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)





# -----------------------------
# Run visualization
# -----------------------------
predict_and_visualize(model, val_loader, device)


