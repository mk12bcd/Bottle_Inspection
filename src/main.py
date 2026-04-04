import cv2
import numpy as np
from picamera2 import Picamera2
import joblib
import time

# ======================
# Load OpenCV DNN model
# ======================
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

# ======================
# Load classifier
# ======================
clf = joblib.load("bottle_classifier.pkl")

label_map = {
    0: "GOOD",
    1: "DEFECTIVE - Missing Cap",
    2: "DEFECTIVE - No Label"
}

# ======================
# Camera setup
# ======================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    buffer_count=2
)
picam2.configure(config)
picam2.start()
time.sleep(0.5)

# ======================
# Counters
# ======================
good_count = 0
defective_count = 0
total_count = 0
DIST_THRESHOLD = 50  # Prevent double-counting
last_counted_box = None

# ======================
# Detection with proper YOLOv8 decoding
# ======================
def detect(frame, conf_threshold=0.25, iou_threshold=0.45):
    h, w = frame.shape[:2]

    # Preprocess image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0]  # Shape: (N, 85)

    boxes = []
    confidences = []
    class_ids = []

    for pred in outputs:
        obj_conf = pred[4]
        class_scores = pred[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = class_scores[class_id]

        # Only detect bottles (COCO class 39)
        if class_id != 39:
            continue

        score = obj_conf * class_conf
        if score < conf_threshold:
            continue

        x_center, y_center, bw, bh = pred[0:4]
        x_center *= w
        y_center *= h
        bw *= w
        bh *= h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h] for NMS
        confidences.append(float(score))
        class_ids.append(class_id)

    # Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    result_boxes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w_box, h_box = boxes[i]
            result_boxes.append((x, y, x + w_box, y + h_box, confidences[i]))

    return result_boxes

# ======================
# Main loop
# ======================
current_bottle_id = None
detection_start_time = None
predictions_buffer = []
CLASSIFICATION_DELAY = 0.7

print("="*50)
print("MKY Automation - Bottle Inspection System")
print("Press 'q' to quit, 'r' to reset counters")
print("="*50)

while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape

    boxes = detect(frame)

    # Find bottle closest to center
    center_x = w // 2
    center_y = h // 2
    best_box = None
    min_distance = float("inf")
    for box in boxes:
        x1, y1, x2, y2, conf = box
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2
        distance = ((box_center_x - center_x)**2 + (box_center_y - center_y)**2)**0.5
        if distance < min_distance:
            min_distance = distance
            best_box = box

    current_time = time.time()

    if best_box:
        x1, y1, x2, y2, conf = best_box
        bottle_id = f"{x1//30}_{y1//30}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # New bottle detected
        if current_bottle_id != bottle_id:
            current_bottle_id = bottle_id
            detection_start_time = current_time
            predictions_buffer = []

        # Show analyzing delay
        if detection_start_time and (current_time - detection_start_time) < CLASSIFICATION_DELAY:
            remaining = int(CLASSIFICATION_DELAY - (current_time - detection_start_time))
            cv2.putText(frame, f"ANALYZING... ({remaining}s)", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        elif detection_start_time and (current_time - detection_start_time) >= CLASSIFICATION_DELAY:
            bottle_roi = frame[y1:y2, x1:x2]
            if bottle_roi.size != 0:
                gray = cv2.cvtColor(bottle_roi, cv2.COLOR_RGB2GRAY)
                gray = cv2.resize(gray, (200, 200))
                features = gray.flatten().reshape(1, -1)

                prediction = clf.predict(features)[0]
                probs = clf.predict_proba(features)[0]
                confidence = max(probs) * 100

                predictions_buffer.append(prediction)
                if len(predictions_buffer) > 3:
                    predictions_buffer.pop(0)

                if predictions_buffer:
                    final_pred = max(set(predictions_buffer), key=predictions_buffer.count)

                    if final_pred == 0:
                        label = f"GOOD ({confidence:.1f}%)"
                        color = (0, 255, 0)
                    else:
                        label = f"{label_map[final_pred]} ({confidence:.1f}%)"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Distance-based counting to prevent duplicates
                    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    count_bottle = True
                    if last_counted_box is not None:
                        last_center = ((last_counted_box[0] + last_counted_box[2]) // 2,
                                       (last_counted_box[1] + last_counted_box[3]) // 2)
                        distance = ((box_center[0] - last_center[0])**2 +
                                    (box_center[1] - last_center[1])**2)**0.5
                        if distance < DIST_THRESHOLD:
                            count_bottle = False

                    if count_bottle:
                        total_count += 1
                        if final_pred == 0:
                            good_count += 1
                        else:
                            defective_count += 1
                        last_counted_box = (x1, y1, x2, y2)

    else:
        current_bottle_id = None
        detection_start_time = None
        predictions_buffer = []
        cv2.putText(frame, "NO BOTTLE", (w//2 - 60, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # UI display
    cv2.putText(frame, "MKY AUTOMATION", (w//2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"TOTAL: {total_count}", (w - 150, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"GOOD: {good_count}", (w - 150, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"DEFECTIVE: {defective_count}", (w - 150, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Center marker
    cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 255, 0), 1)
    cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 255, 0), 1)
    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 0), -1)

    cv2.imshow("MKY Automation - Bottle Inspection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        good_count = 0
        defective_count = 0
        total_count = 0
        last_counted_box = None
        print("Counters reset!")

cv2.destroyAllWindows()
picam2.close()

print("\n" + "="*50)
print("MKY AUTOMATION - FINAL REPORT")
print(f"Total Bottles: {total_count}")
print(f"Good Bottles: {good_count}")
print(f"Defective Bottles: {defective_count}")
print("="*50)