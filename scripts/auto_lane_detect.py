import cv2
import numpy as np
from pathlib import Path

# Load video and get first frame
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")
cap = cv2.VideoCapture(str(DATA_ROOT / "videos/traffic3.mp4"))
success, frame = cap.read()
if not success:
    print("Could not read video.")
    cap.release()
    exit()

# Resize frame
TARGET_WIDTH = 1280
scale_ratio = TARGET_WIDTH / frame.shape[1]
frame = cv2.resize(frame, (TARGET_WIDTH, int(frame.shape[0] * scale_ratio)))

mask = np.zeros_like(frame[:, :, 0])  # same size, single channel

# Define a polygon covering the roads (adjust this as needed!)
road_polygon = np.array([
    [200, 500], [1080, 500],
    [1280, 720], [0, 720]
])

# Fill polygon with white (area to keep)
cv2.fillPoly(mask, [road_polygon], 255)

# Apply grayscale, blur, and Canny edge detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

masked_edges = cv2.bitwise_and(edges, mask)

# Hough Line Transform
lines = cv2.HoughLinesP(
    masked_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=100,
    minLineLength=100,
    maxLineGap=40
)

# Draw lines on a copy of original
line_image = frame.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show result
cv2.imshow("Masked Lane Detection", line_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
