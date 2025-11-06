# lane_flow_analyzer.py
# Optical-flow lane analytics + YOLOv8 lane-filtered counting.
# All arguments are set in the CONFIG section below.

import os
import cv2
import csv
import gc
import math
import numpy as np
from typing import List, Tuple, Optional

# ===================== CONFIG (edit here) =====================
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR        = str(ROOT / "outputs/video-frames")
MASKS_DIR         = str(ROOT / "outputs/masks")          # binary masks (0/255) from lane_mask_generator
OUTPUT_CSV        = str(ROOT / "outputs/analytics/flow_metrics.csv")
ANNOTATED_VIDEO   = str(ROOT / "outputs/analytics/flow_annotated.mp4")  # set "" to skip video
MAKE_VIDEO        = True
FPS_ASSUME        = 30.0  # used for video writer timing if MAKE_VIDEO

# Lane ROI from mask is often thin (lane markings). We dilate to get a usable band region.
DILATE_KERNEL     = (17, 17)   # widen ROI band if masks are sparse
DILATE_ITER       = 2

# Optical flow parameters (Farnebäck)
FB_PYR_SCALE      = 0.5
FB_LEVELS         = 3
FB_WINSIZE        = 21
FB_ITERATIONS     = 3
FB_POLY_N         = 5
FB_POLY_SIGMA     = 1.2
FB_FLAGS          = 0

# Motion magnitude thresholds (pixels/frame)
MOTION_THRESH     = 1.2   # counts as "moving" if |flow| >= this
SAT_CLIP_PCT      = 99.0  # clip flow magnitude at this percentile for stability

# Drawing (sparse arrows for readability)
DRAW_EVERY        = 12     # sample grid step for arrows
ARROW_SCALE       = 3.0    # arrow length scaling
FONT              = cv2.FONT_HERSHEY_SIMPLEX

# ================= YOLOv8 (lane-filtered counting) ==============
# Requires: pip install ultralytics
YOLO_MODEL_PATH   = str(ROOT / "models/yolo/yolov8n.pt")  # set to your .pt, or leave as is to try default
YOLO_CONF         = 0.35   # confidence threshold
YOLO_IOU          = 0.45   # NMS IoU threshold
ROI_OVERLAP_MIN   = 0.20   # min % of bbox area overlapping lane ROI to count

# COCO class IDs used for counting
COCO = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
    "bus": 5, "truck": 7
}
COUNT_CLASSES = {
    "vehicle": [COCO["car"], COCO["truck"], COCO["bus"], COCO["motorcycle"], COCO["bicycle"]],
    "person": [COCO["person"]],
}
# ===============================================================


# -------------------- Utilities --------------------
def list_images_sorted(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    names = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    names.sort()
    return names

def ensure_dirs():
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if MAKE_VIDEO and ANNOTATED_VIDEO:
        out_vdir = os.path.dirname(ANNOTATED_VIDEO)
        if out_vdir and not os.path.exists(out_vdir):
            os.makedirs(out_vdir, exist_ok=True)

def to_gray_robust(arr: np.ndarray) -> np.ndarray:
    """
    Convert input array to 2D grayscale robustly:
    - HxW (2D): return as-is
    - HxWx1: squeeze last channel
    - HxWx(3|4): cvtColor BGR/RGBA -> GRAY
    """
    if arr is None:
        return arr
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            return arr[:, :, 0]
        elif c >= 3:
            # Assume BGR[A]
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    # Fallback: try squeeze
    return np.squeeze(arr)

def normalize_mask_0_255(mask_any: np.ndarray) -> np.ndarray:
    """
    Ensure mask is uint8 with values {0,255}.
    Accepts float in [0,1], uint8 {0,1}, or uint8 {0,255}, etc.
    """
    if mask_any is None:
        return mask_any
    m = mask_any
    if m.dtype != np.uint8:
        # If float or other types: scale
        m = m.astype(np.float32)
        if m.max() <= 1.0:
            m = (m * 255.0)
        m = np.clip(m, 0, 255).astype(np.uint8)
    # If it's mostly 0/1, scale up
    if 0 < m.max() <= 1:
        m = (m.astype(np.uint8) * 255)
    return m

def make_lane_roi(mask_any: np.ndarray) -> np.ndarray:
    """Take a predicted mask (any shape/dtype), make 0/255 grayscale, then dilate to build a lane-band ROI."""
    mask = to_gray_robust(mask_any)
    mask = normalize_mask_0_255(mask)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATE_KERNEL)
    roi = cv2.dilate(mask_bin, kernel, iterations=DILATE_ITER)
    return roi  # 0/255

def compute_optical_flow(prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (flow, mag) where flow is HxWx2, mag is HxW."""
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray, next=gray, flow=None,
        pyr_scale=FB_PYR_SCALE, levels=FB_LEVELS, winsize=FB_WINSIZE,
        iterations=FB_ITERATIONS, poly_n=FB_POLY_N, poly_sigma=FB_POLY_SIGMA, flags=FB_FLAGS
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return flow, mag

def clip_by_percentile(arr: np.ndarray, pct: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    hi = np.percentile(arr, pct)
    return np.minimum(arr, hi)

def draw_flow_arrows(frame_bgr: np.ndarray, flow: np.ndarray, roi_mask: np.ndarray):
    H, W = frame_bgr.shape[:2]
    for y in range(0, H, DRAW_EVERY):
        for x in range(0, W, DRAW_EVERY):
            if roi_mask[y, x] == 0:
                continue
            dx, dy = flow[y, x]
            end_x = int(x + ARROW_SCALE * dx)
            end_y = int(y + ARROW_SCALE * dy)
            cv2.arrowedLine(frame_bgr, (x, y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.3)

def compute_metrics(mag_in_roi: np.ndarray, roi_mask: np.ndarray) -> dict:
    roi_area = int(np.count_nonzero(roi_mask))
    if roi_area == 0:
        return {
            "flow_mag_mean": 0.0,
            "flow_mag_median": 0.0,
            "motion_density": 0.0,
            "roi_area_px": 0,
            "queue_index": 0.0
        }
    mag_flat = mag_in_roi[roi_mask > 0]
    mag_flat = clip_by_percentile(mag_flat, SAT_CLIP_PCT)

    flow_mag_mean = float(np.mean(mag_flat)) if mag_flat.size else 0.0
    flow_mag_median = float(np.median(mag_flat)) if mag_flat.size else 0.0
    motion_density = float(np.mean(mag_flat >= MOTION_THRESH)) if mag_flat.size else 0.0

    queue_index = (1.0 - motion_density) * math.log1p(roi_area / 1e4)

    return {
        "flow_mag_mean": flow_mag_mean,
        "flow_mag_median": flow_mag_median,
        "motion_density": motion_density,
        "roi_area_px": roi_area,
        "queue_index": float(queue_index),
    }

# -------------------- YOLOv8 integration --------------------
class YoloWrapper:
    def __init__(self, model_path: str, conf: float, iou: float):
        self.enabled = False
        self.model = None
        self.names = {}
        try:
            from ultralytics import YOLO  # lazy import
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                self.model = YOLO("yolov8n.pt")
            self.names = self.model.names
            self.model.overrides["conf"] = conf
            self.model.overrides["iou"] = iou
            self.enabled = True
            print("[YOLO] Loaded:", getattr(self.model, "model", "model"))
        except Exception as e:
            print(f"[YOLO] Disabled: {e}")

    def detect(self, frame_bgr: np.ndarray):
        """Return list of (xyxy, cls_id, conf) in image coordinates."""
        if not self.enabled:
            return []
        from ultralytics.utils.torch_utils import smart_inference_mode
        with smart_inference_mode():
            results = self.model.predict(source=frame_bgr[..., ::-1], verbose=False)  # BGR->RGB
        dets = []
        if not results:
            return dets
        r0 = results[0]
        if r0.boxes is None or r0.boxes.data is None:
            return dets
        boxes = r0.boxes.xyxy.cpu().numpy()
        clses = r0.boxes.cls.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy()
        for b, c, s in zip(boxes, clses, confs):
            x1, y1, x2, y2 = b
            dets.append(((int(x1), int(y1), int(x2), int(y2)), int(c), float(s)))
        return dets

def bbox_roi_overlap(mask: np.ndarray, box: Tuple[int,int,int,int]) -> float:
    """Return fraction of bbox area covered by ROI mask (mask>0)."""
    H, W = mask.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2))
    y2 = max(0, min(H,   y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = mask[y1:y2, x1:x2]
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    overlap = float(np.count_nonzero(sub))
    return overlap / float(area)

def count_in_roi(roi_mask: np.ndarray, dets: List[Tuple[Tuple[int,int,int,int], int, float]],
                 classes_by_bucket: dict, overlap_min: float):
    counts = {k: 0 for k in classes_by_bucket.keys()}
    per_class = {}
    for k in classes_by_bucket:
        per_class[k] = {}
    for (box, cls_id, conf) in dets:
        frac = bbox_roi_overlap(roi_mask, box)
        if frac < overlap_min:
            continue
        for bucket, ids in classes_by_bucket.items():
            if cls_id in ids:
                counts[bucket] += 1
                per_class[bucket][cls_id] = per_class[bucket].get(cls_id, 0) + 1
    return counts, per_class

def draw_detections(frame_bgr: np.ndarray, dets, roi_mask: np.ndarray, names: dict):
    for (box, cls_id, conf) in dets:
        x1, y1, x2, y2 = box
        frac = bbox_roi_overlap(roi_mask, box)
        color = (0, 255, 0) if frac >= ROI_OVERLAP_MIN else (128, 128, 128)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cls_id, str(cls_id))}:{conf:.2f}"
        cv2.putText(frame_bgr, label, (x1, max(0, y1-5)), FONT, 0.5, color, 1, cv2.LINE_AA)

# -------------------- Main --------------------
def run():
    ensure_dirs()

    frame_names = list_images_sorted(FRAMES_DIR)
    mask_names  = list_images_sorted(MASKS_DIR)
    if not frame_names:
        raise FileNotFoundError(f"No frames found in {FRAMES_DIR}")
    if not mask_names:
        raise FileNotFoundError(f"No masks found in {MASKS_DIR}")

    # Match by index if counts differ
    N = min(len(frame_names), len(mask_names))
    frame_names = frame_names[:N]
    mask_names  = mask_names[:N]

    # Video writer
    writer = None
    if MAKE_VIDEO and ANNOTATED_VIDEO:
        first_frame = cv2.imread(os.path.join(FRAMES_DIR, frame_names[0]))
        if first_frame is None:
            raise RuntimeError("Could not read first frame for video sizing.")
        H, W = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(ANNOTATED_VIDEO, fourcc, FPS_ASSUME, (W, H))

    # CSV setup
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    csv_file = open(OUTPUT_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_idx", "frame_name",
        "flow_mag_mean", "flow_mag_median", "motion_density",
        "roi_area_px", "queue_index",
        "veh_count", "ped_count",
        "car", "truck", "bus", "motorcycle", "bicycle"
    ])

    # YOLO
    yolo = YoloWrapper(YOLO_MODEL_PATH, YOLO_CONF, YOLO_IOU)

    prev_gray = None
    saved_rows = 0

    for i in range(N):
        frame_path = os.path.join(FRAMES_DIR, frame_names[i])
        mask_path  = os.path.join(MASKS_DIR,  mask_names[i])

        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        # Read mask permissively: could be 1, 3, or 4 channels (PNG)
        mask_any = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if frame is None or mask_any is None:
            print(f"[Skip] unreadable pair: {frame_names[i]} / {mask_names[i]}")
            continue

        if frame.shape[:2] != mask_any.shape[:2]:
            mask_any = cv2.resize(mask_any, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        roi = make_lane_roi(mask_any)  # 0/255 robust
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow (vs previous frame)
        if prev_gray is None:
            metrics = {
                "flow_mag_mean": 0.0,
                "flow_mag_median": 0.0,
                "motion_density": 0.0,
                "roi_area_px": int(np.count_nonzero(roi)),
                "queue_index": 0.0
            }
        else:
            try:
                flow, mag = compute_optical_flow(prev_gray, gray)
            except Exception as e:
                print(f"[Flow ERR] {frame_names[i]}: {e}")
                prev_gray = gray
                del frame, mask_any, roi, gray
                gc.collect()
                continue
            mag_in_roi = np.where(roi > 0, mag, 0.0)
            metrics = compute_metrics(mag_in_roi, roi)

        # YOLO detections + lane filtering
        dets = yolo.detect(frame) if yolo.enabled else []
        counts, per_class = count_in_roi(roi, dets, COUNT_CLASSES, ROI_OVERLAP_MIN)
        veh_count = int(counts.get("vehicle", 0))
        ped_count = int(counts.get("person", 0))

        car_n        = per_class.get("vehicle", {}).get(COCO["car"], 0)
        truck_n      = per_class.get("vehicle", {}).get(COCO["truck"], 0)
        bus_n        = per_class.get("vehicle", {}).get(COCO["bus"], 0)
        motorcycle_n = per_class.get("vehicle", {}).get(COCO["motorcycle"], 0)
        bicycle_n    = per_class.get("vehicle", {}).get(COCO["bicycle"], 0)

        # Annotate and draw
        if writer is not None:
            frame_draw = frame.copy()
            if prev_gray is not None:
                draw_flow_arrows(frame_draw, flow, roi)
            if yolo.enabled:
                draw_detections(frame_draw, dets, roi, yolo.names)

            hud1 = (f"mean|v|={metrics['flow_mag_mean']:.2f}  "
                    f"med|v|={metrics['flow_mag_median']:.2f}  "
                    f"mov_dens={metrics['motion_density']:.2f}  "
                    f"queue_idx={metrics['queue_index']:.2f}")
            hud2 = (f"veh={veh_count} (car:{car_n} truck:{truck_n} bus:{bus_n} moto:{motorcycle_n} bike:{bicycle_n})  "
                    f"ped={ped_count}")
            cv2.rectangle(frame_draw, (8, 8), (8 + 1100, 8 + 56), (0, 0, 0), -1)
            cv2.putText(frame_draw, hud1, (16, 30), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_draw, hud2, (16, 54), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame_draw)
            del frame_draw

        # CSV row
        csv_writer.writerow([
            i, frame_names[i],
            metrics["flow_mag_mean"], metrics["flow_mag_median"], metrics["motion_density"],
            metrics["roi_area_px"], metrics["queue_index"],
            veh_count, ped_count,
            car_n, truck_n, bus_n, motorcycle_n, bicycle_n
        ])
        saved_rows += 1

        # Advance
        prev_gray = gray

        # Cleanup
        if prev_gray is not None and 'flow' in locals():
            del flow, mag, mag_in_roi
        del frame, mask_any, roi, gray, metrics, dets, counts, per_class
        gc.collect()

    csv_file.close()
    if writer is not None:
        writer.release()

    print(f"[DONE] Saved {saved_rows} rows → {OUTPUT_CSV}")
    if writer is not None and ANNOTATED_VIDEO:
        print(f"[DONE] Annotated video → {ANNOTATED_VIDEO}")


if __name__ == "__main__":
    run()
