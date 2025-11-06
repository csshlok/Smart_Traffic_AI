import cv2
import numpy as np
from pathlib import Path

# Load your video
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")
video_path = DATA_ROOT / "videos/traffic.mp4"
cap = cv2.VideoCapture(str(video_path))

success, frame = cap.read()
cap.release()

if not success:
    print("Failed to read video")
    exit()

# Resize for visibility (optional)
frame = cv2.resize(frame, (1280, 720))

# Storage for points
points = []

# Mouse callback function
def draw_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

cv2.namedWindow("Draw ROI - Press Enter to Save")
cv2.setMouseCallback("Draw ROI - Press Enter to Save", draw_point)

while True:
    temp_frame = frame.copy()
    for pt in points:
        cv2.circle(temp_frame, pt, 5, (0, 255, 0), -1)

    if len(points) > 1:
        cv2.polylines(temp_frame, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)

    cv2.imshow("Draw ROI - Press Enter to Save", temp_frame)
    key = cv2.waitKey(1)

    if key == 13:  # ENTER key
        break
    elif key == 27:  # ESC key to cancel
        points = []

cv2.destroyAllWindows()

# Save ROI coordinates
roi_name = input("Name this ROI (e.g., lane1): ")
roi_dir = DATA_ROOT / "roi"
roi_dir.mkdir(parents=True, exist_ok=True)
roi_path = roi_dir / f"{roi_name}_roi.npy"
np.save(str(roi_path), np.array(points))
print(f"Saved ROI to {roi_path}")
