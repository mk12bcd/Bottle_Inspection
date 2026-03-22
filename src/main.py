# main.py

from picamera2 import Picamera2
import cv2
import joblib
import numpy as np
from ultralytics import YOLO

# ======================
# Load classifier
# ======================
clf = joblib.load("bottle_classifier.pkl")

label_map = {
    0: "GOOD BOTTLE",
    1: "MISSING CAP",
    2: "NO LABEL"
}

# ======================
# Load YOLO
# ======================
yolo_model = YOLO("yolov8n.pt")

# ======================
# Camera
# ======================
picam2 = Picamera2()
picam2.start()

# ======================
# Stability buffers
# ======================
predictions_buffer = []
stable_count = 0

print("Running... Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape

    # Invisible detection zone (center area)
    zone_x1 = int(w * 0.2)
    zone_y1 = int(h * 0.1)
    zone_x2 = int(w * 0.8)
    zone_y2 = int(h * 0.9)

    results = yolo_model(frame, imgsz=320)  # smaller size = faster

    bottle_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 39 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Check if inside zone
                if (x1 > zone_x1 and y1 > zone_y1 and x2 < zone_x2 and y2 < zone_y2):
                    bottle_detected = True

                    # Crop bottle
                    bottle = frame[y1:y2, x1:x2]
                    if bottle.size == 0:
                        continue

                    # Preprocess
                    gray = cv2.cvtColor(bottle, cv2.COLOR_BGR2GRAY)
                    gray_resized = cv2.resize(gray, (200, 200))
                    features = gray_resized.flatten().reshape(1, -1)

                    # Predict
                    prediction = clf.predict(features)[0]

                    # Stability logic
                    predictions_buffer.append(prediction)
                    if len(predictions_buffer) > 5:
                        predictions_buffer.pop(0)

                    stable_count += 1

                    if stable_count > 5:
                        final_pred = max(set(predictions_buffer), key=predictions_buffer.count)
                        label = label_map[final_pred]

                        color = (0,255,0) if final_pred == 0 else (0,0,255)

                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                else:
                    stable_count = 0
                    predictions_buffer.clear()

    if not bottle_detected:
        stable_count = 0
        predictions_buffer.clear()
        cv2.putText(frame, "NO BOTTLE", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Bottle Inspection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()