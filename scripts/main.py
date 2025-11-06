import cv2
from ultralytics import YOLO
from supervision.draw.color import Color
import numpy as np
from supervision import Detections
from supervision.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator
from pathlib import Path

TARGET_WIDTH = 1920
original_width = None
original_height = None
scaling_ratio = None


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("T:/data")

model = YOLO(str(ROOT / "runs/detect/train4/weights/best.pt"))
cap = cv2.VideoCapture(str(DATA_ROOT / "videos/traffic5.mp4"))
# Get original video size
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate scaling ratio based on width
scaling_ratio = TARGET_WIDTH / original_width
target_height = int(original_height * scaling_ratio)


vehicles = ["car", "truck", "bus", "motorcycle"]
pedestrians = ["person"]

#lane1_polygon = np.load(str(DATA_ROOT / "roi/lane1_roi.npy"))
#lane2 = np.load(str(DATA_ROOT / "roi/lane2_roi.npy"))
#crosswalk_polygon = np.load(str(DATA_ROOT / "roi/crosswalk_roi.npy"))

#zone_lane1 = PolygonZone(polygon=lane1_polygon)
#zone_lane2 = PolygonZone(polygon=lane2)
#zone_crosswalk = PolygonZone(polygon=crosswalk_polygon)
#zone_annotator_lane1 = PolygonZoneAnnotator(zone=zone_lane1, color=Color.from_rgb_tuple((255, 0, 0)), thickness=2)
#zone_annotator_lane2 = PolygonZoneAnnotator(zone=zone_lane2, color=Color.from_rgb_tuple((0, 255, 0)), thickness=2)
#zone_annotator_crosswalk = PolygonZoneAnnotator(zone=zone_crosswalk, color=Color.from_rgb_tuple((0, 0, 255)), thickness=2)


while cap.isOpened():
    success, frame = cap.read()  # get the next frame
    if not success:
        break  # if no more video, stop the loop

    frame = cv2.resize(frame, (TARGET_WIDTH, target_height))

    results = model(frame)[0]  # [0] gives the first set of results
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()

    # Convert to Supervision format
    detections = Detections(
        xyxy=boxes,
        class_id=classes,
        confidence=confidences
    )

    #lane1_count = zone_lane1.trigger(detections=detections).sum()
    #lane2_count = zone_lane2.trigger(detections=detections).sum()
    #crosswalk_count = zone_crosswalk.trigger(detections=detections).sum()

    # Annotate zones
    #frame = zone_annotator_lane1.annotate(frame)
    #frame = zone_annotator_lane2.annotate(frame)
    #frame = zone_annotator_crosswalk.annotate(frame)

    # Draw text labels with counts
    #cv2.putText(frame, f"Lane 1: {lane1_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #cv2.putText(frame, f"Lane 2: {lane2_count}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.putText(frame, f"Crosswalk: {crosswalk_count}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    for box in results.boxes:
        class_id = int(box.cls[0])  # get what kind of object it is (car/person)
        label = results.names[class_id]  # get the name like "car" or "person"
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # get box corners
        confidence = float(box.conf[0])  # how sure is the AI

        if label in vehicles:
            color = Color.BLUE  # cars = blue boxes
        elif label in pedestrians:
            color = Color.GREEN  # people = green boxes
        else:
            continue  # ignore animals or signs

        cv2.rectangle(frame, (x1, y1), (x2, y2), color.as_bgr(), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.as_bgr(), 1)

    cv2.imshow("Traffic Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
