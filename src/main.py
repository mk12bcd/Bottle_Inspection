
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
    0: "Good Bottles",
    1: "Defective Bottles - Missing Cap",
    2: "Defective Bottles - No Label"
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
# Counters
# ======================
good_count = 0
defective_count = 0
last_counted = False

# ======================
# Stability
# ======================
predictions_buffer = []
confidence_buffer = []
stable_count = 0

print("Running... Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape

    # ======================
    # Invisible detection zone
    # ======================
    zone_x1 = int(w * 0.2)
    zone_y1 = int(h * 0.1)
    zone_x2 = int(w * 0.8)
    zone_y2 = int(h * 0.9)

    results = yolo_model(frame, imgsz=320)

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

                    # ======================
                    # Preprocess
                    # ======================
                    gray = cv2.cvtColor(bottle, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (200, 200))
                    features = gray.flatten().reshape(1, -1)

                    # ======================
                    # Predict + Confidence
                    # ======================
                    prediction = clf.predict(features)[0]
                    probs = clf.predict_proba(features)[0]
                    confidence = max(probs) * 100

                    # ======================
                    # Smoothing buffers
                    # ======================
                    predictions_buffer.append(prediction)
                    confidence_buffer.append(confidence)

                    if len(predictions_buffer) > 5:
                        predictions_buffer.pop(0)
                        confidence_buffer.pop(0)

                    stable_count += 1

                    # ======================
                    # Stable decision
                    # ======================
                    if stable_count > 3:
                        final_pred = max(set(predictions_buffer), key=predictions_buffer.count)
                        final_conf = sum(confidence_buffer) / len(confidence_buffer)

                        # ======================
                        # Label formatting
                        # ======================
                        if final_pred == 0:
                            label = f"GOOD BOTTLE ({final_conf:.1f}%)"
                            color = (0, 255, 0)
                        else:
                            defect = label_map[final_pred].split("-")[-1].strip()
                            label = f"DEFECTIVE - {defect} ({final_conf:.1f}%)"
                            color = (0, 0, 255)

                        # ======================
                        # Counting logic
                        # ======================
                        if not last_counted:
                            if final_pred == 0:
                                good_count += 1
                            else:
                                defective_count += 1
                            last_counted = True

                        # ======================
                        # Draw results
                        # ======================
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                else:
                    stable_count = 0
                    predictions_buffer.clear()
                    confidence_buffer.clear()

    # ======================
    # No bottle detected
    # ======================
    if not bottle_detected:
        stable_count = 0
        predictions_buffer.clear()
        confidence_buffer.clear()
        last_counted = False

        cv2.putText(frame, "NO BOTTLE", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # ======================
    # Display counters
    # ======================
    cv2.putText(frame, f"GOOD: {good_count}", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"DEFECTIVE: {defective_count}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Bottle Inspection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()