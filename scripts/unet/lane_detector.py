import cv2
import torch
import numpy as np
from traffic_ai.unet.model import UNet
from traffic_ai.unet.dataset_loader import preprocess_image_only
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from traffic_ai.unet.attention_model import Attention_UNet
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")
VIDEO_PATH = str(DATA_ROOT / "videos/traffic5.mp4")
OUTPUT_PATH = str(ROOT / "outputs/lane_output_video.mp4")
MODEL_PATH = str(ROOT / "models/unet/attention_unet_checkpoint.pth")
IMAGE_SIZE = (256, 256)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform for inference
transform = A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Load model
model = Attention_UNet(in_channels=4, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Read input video
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Frame-by-frame inference
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_mask = model(image_tensor)
        pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        pred_mask = cv2.resize(pred_mask, (width, height))

    # Apply mask overlay (green lanes)
    colored_mask = np.zeros_like(orig)
    colored_mask[:, :, 1] = pred_mask  # green channel
    overlay = cv2.addWeighted(orig, 1.0, colored_mask, 0.5, 0)

    out.write(overlay)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Lane overlay video saved to: {OUTPUT_PATH}")
