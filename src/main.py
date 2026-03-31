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
cooldown = 0

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

    # Slight blur reduction (optional)
    frame = cv2.GaussianBlur(frame, (3,3), 0)

    # Title
    cv2.putText(frame, "Bottle Inspection System", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Center marker
    cv2.circle(frame, (w//2, h//2), 5, (255,255,0), -1)

    # ======================
    # YOLO Detection
    # ======================
    results = yolo_model(frame, imgsz=320)

    bottle_detected = False
    best_box = None
    min_distance = float("inf")

    center_x = w // 2
    center_y = h // 2

    # ======================
    # Find closest bottle
    # ======================
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 39 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                box_cx = (x1 + x2) // 2
                box_cy = (y1 + y2) // 2

                dist = ((box_cx - center_x)**2 + (box_cy - center_y)**2)**0.5

                if dist < min_distance:
                    min_distance = dist
                    best_box = (x1, y1, x2, y2)
                    bottle_detected = True

    # ======================
    # If bottle found
    # ======================
    if bottle_detected and best_box is not None:
        x1, y1, x2, y2 = best_box

        bottle = frame[y1:y2, x1:x2]
        if bottle.size != 0:

            # Preprocess
            gray = cv2.cvtColor(bottle, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))
            features = gray.flatten().reshape(1, -1)

            # Predict
            prediction = clf.predict(features)[0]
            probs = clf.predict_proba(features)[0]
            confidence = max(probs) * 100

            # Buffers
            predictions_buffer.append(prediction)
            confidence_buffer.append(confidence)

            if len(predictions_buffer) > 3:
                predictions_buffer.pop(0)
                confidence_buffer.pop(0)

            stable_count += 1
            
            # ========== FIX #1: Decrement cooldown when bottle present ==========
            if cooldown > 0:
                cooldown -= 1

            # Detecting state
            if stable_count <= 1:
                cv2.putText(frame, "DETECTING...", (30,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # Final decision
            if stable_count > 1:
                final_pred = max(set(predictions_buffer), key=predictions_buffer.count)
                final_conf = sum(confidence_buffer) / len(confidence_buffer)

                if final_pred == 0:
                    label = f"GOOD BOTTLE ({final_conf:.1f}%)"
                    color = (0,255,0)
                else:
                    defect = label_map[final_pred].split("-")[-1].strip()
                    label = f"DEFECTIVE - {defect} ({final_conf:.1f}%)"
                    color = (0,0,255)

                # ========== FIX #2: Reset last_counted when cooldown is done ==========
                if not last_counted and cooldown <= 0:
                    if final_pred == 0:
                        good_count += 1
                    else:
                        defective_count += 1

                    last_counted = True
                    cooldown = 10

                # Draw
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    else:
        # Reset when no bottle
        stable_count = 0
        predictions_buffer.clear()
        confidence_buffer.clear()
        # ========== FIX #3: Reset last_counted when bottle leaves ==========
        last_counted = False
        # Don't reset cooldown here - let it continue counting down

        cv2.putText(frame, "NO BOTTLE", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # ======================
    # Counters display (TOP RIGHT)
    # ======================
    cv2.putText(frame, f"GOOD: {good_count}", (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"DEFECTIVE: {defective_count}", (w - 180, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Bottle Inspection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()