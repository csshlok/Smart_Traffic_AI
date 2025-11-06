# lane_mask_generator.py — True Tiling, Seamless Blending, OOM-Resilient
import os
import gc
import cv2
import math
import torch
import numpy as np
from typing import Tuple, Iterator
from traffic_ai.unet.attention_model import Attention_UNet  # __init__(in_channels=3, out_channels=1)
from pathlib import Path

# =================== USER CONFIG ===================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT / "models/unet/attention_unet_checkpoint.pth")

INPUT_FRAMES_DIR    = str(ROOT / "outputs/video-frames")
OUTPUT_MASKS_DIR    = str(ROOT / "outputs/masks")
OUTPUT_OVERLAYS_DIR = str(ROOT / "outputs/overlays")

# Tile size should be a multiple of 32 for U-Net-like enc/decoders
TILE_SIZE   = (512, 512)   # (width, height)
TILE_OVERLAP = 64          # pixels of overlap (both directions)
THRESH = 0.5               # sigmoid threshold
SAVE_OVERLAYS = True
# ===================================================


# -------------------- Utilities --------------------
def setup_backend():
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True


def load_model(model_path: str, device: str) -> torch.nn.Module:
    model = Attention_UNet(in_channels=3, out_channels=1)
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    if device == "cuda":
        model.to("cuda")
        model = model.to(memory_format=torch.channels_last)
        model.half()  # FP16 on GPU
    return model


def make_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """
    Create a smooth blending weight map for a tile of size (h,w).
    We taper weights towards edges over 'overlap' pixels using a raised-cosine.
    """
    eps = 1e-3
    def ramp(n):
        # 0..1 raised-cosine over n samples; if n<=0, return empty
        if n <= 0: return np.array([], dtype=np.float32)
        t = np.linspace(0, math.pi, n, dtype=np.float32)
        return (1 - np.cos(t)) * 0.5  # 0..1

    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)

    n = min(overlap, h // 2)
    m = min(overlap, w // 2)
    if n > 0:
        up = ramp(n)             # 0..1
        dn = up[::-1]            # 1..0
        wy[:n] *= np.maximum(up, eps)
        wy[-n:] *= np.maximum(dn, eps)
    if m > 0:
        up = ramp(m)
        dn = up[::-1]
        wx[:m] *= np.maximum(up, eps)
        wx[-m:] *= np.maximum(dn, eps)

    weight = np.outer(wy, wx)  # (h,w)
    return weight


def tiles(H: int, W: int, tile_w: int, tile_h: int, overlap: int) -> Iterator[Tuple[int,int,int,int]]:
    """
    Yield tile boxes (x0, y0, x1, y1) covering the full image with given overlap.
    Tiles at borders will be clamped to image bounds.
    """
    step_x = tile_w - overlap
    step_y = tile_h - overlap
    if step_x <= 0 or step_y <= 0:
        raise ValueError("TILE_OVERLAP must be smaller than TILE_SIZE in both dimensions.")
    y = 0
    while True:
        x = 0
        y1 = min(y + tile_h, H)
        y0 = max(0, y1 - tile_h)
        last_row = (y1 == H)
        while True:
            x1 = min(x + tile_w, W)
            x0 = max(0, x1 - tile_w)
            yield (x0, y0, x1, y1)
            if x1 == W:
                break
            x += step_x
        if last_row:
            break
        y += step_y


def to_tensor_chw(image_bgr: np.ndarray, device: str) -> torch.Tensor:
    """BGR->RGB, HWC uint8 -> CHW float [0,1]; FP16 if CUDA."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0  # CHW float32
    if device == "cuda":
        ten = ten.half().cuda(non_blocking=True)  # 3D tensor; channels_last applied after unsqueeze
    return ten


@torch.inference_mode()
def model_prob_single(model: torch.nn.Module, img_chw: torch.Tensor, device: str) -> np.ndarray:
    """
    Forward pass returning probabilities in [0,1] as numpy float32 (no threshold).
    """
    x = img_chw.unsqueeze(0)  # NCHW 4D
    if device == "cuda":
        x = x.to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            y = model(x)
    else:
        y = model(x)
    prob = torch.sigmoid(y).squeeze(0).squeeze(0).float().cpu().numpy()  # HxW float32
    del x, y
    return prob


def overlay_mask(image_bgr: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask_uint8 > 0] = (0, 255, 0)
    return cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)


# -------------------- Main pipeline (True Tiling) --------------------
def infer_frame_tiled(model: torch.nn.Module, frame_bgr: np.ndarray, device: str) -> np.ndarray:
    """
    Run true-tiling inference on a full-resolution frame and return a binary mask (uint8 0/255).
    """
    H, W = frame_bgr.shape[:2]
    tile_w, tile_h = TILE_SIZE
    prob_accum = np.zeros((H, W), dtype=np.float32)
    wsum_accum  = np.zeros((H, W), dtype=np.float32)

    # Precompute a max-size weight; edge tiles will slice it to match actual tile size
    base_weight = make_weight(tile_h, tile_w, TILE_OVERLAP)

    for (x0, y0, x1, y1) in tiles(H, W, tile_w, tile_h, TILE_OVERLAP):
        tile = frame_bgr[y0:y1, x0:x1]
        h, w = tile.shape[:2]
        weight = base_weight[:h, :w] if (h, w) != base_weight.shape else base_weight

        # Resize tile to model input size if needed (for border tiles smaller than TILE_SIZE)
        if (w, h) != (tile_w, tile_h):
            tile_for_net = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            weight_for_net = base_weight  # keep weight at model size
        else:
            tile_for_net = tile
            weight_for_net = weight

        try:
            ten = to_tensor_chw(tile_for_net, device)
            prob_small = model_prob_single(model, ten, device)  # (tile_h, tile_w)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if device == "cuda" and ("out of memory" in str(e).lower() or "cuda error" in str(e).lower()):
                # Per-tile GPU → CPU fallback
                torch.cuda.empty_cache()
                gc.collect()
                ten_cpu = to_tensor_chw(tile_for_net, "cpu")
                prob_small = model_prob_single(model.cpu(), ten_cpu, "cpu")
                # move model back to cuda for next tiles
                if torch.cuda.is_available():
                    model.to("cuda").to(memory_format=torch.channels_last).half()
            else:
                raise

        # Resize prob back if we up/downscaled for border tiles
        if (w, h) != (tile_w, tile_h):
            prob_tile = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
            weight_tile = weight  # match border tile size
        else:
            prob_tile = prob_small
            weight_tile = weight_for_net

        # Accumulate with weights
        prob_accum[y0:y1, x0:x1] += (prob_tile * weight_tile).astype(np.float32)
        wsum_accum[y0:y1, x0:x1]  += weight_tile.astype(np.float32)

        # Hygiene
        del tile, ten, prob_small, prob_tile, weight_tile
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Normalize and threshold
    prob_full = prob_accum / (wsum_accum + 1e-6)
    mask = (prob_full >= THRESH).astype(np.uint8) * 255
    return mask


def generate_lane_masks():
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
    if SAVE_OVERLAYS:
        os.makedirs(OUTPUT_OVERLAYS_DIR, exist_ok=True)

    setup_backend()
    model = load_model(MODEL_PATH, DEVICE)

    files = sorted(
        f for f in os.listdir(INPUT_FRAMES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    saved = 0
    for fname in files:
        fpath = os.path.join(INPUT_FRAMES_DIR, fname)
        frame = cv2.imread(fpath)
        if frame is None:
            print(f"[Skip] unreadable: {fname}")
            continue

        try:
            mask = infer_frame_tiled(model, frame, DEVICE)
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")
            del frame
            continue

        # Save outputs
        cv2.imwrite(os.path.join(OUTPUT_MASKS_DIR, fname), mask)
        if SAVE_OVERLAYS:
            overlay = overlay_mask(frame, mask)
            cv2.imwrite(os.path.join(OUTPUT_OVERLAYS_DIR, fname), overlay)
            del overlay

        saved += 1

        # Cleanup
        del frame, mask
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"[DONE] Saved {saved} masks → {OUTPUT_MASKS_DIR}")


if __name__ == "__main__":
    generate_lane_masks()
