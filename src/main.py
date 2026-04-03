import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import joblib

# ======================
# Load ONNX model
# ======================
session = ort.InferenceSession("yolov8n.onnx")
input_name = session.get_inputs()[0].name

# ======================
# Load classifier
# ======================
clf = joblib.load("bottle_classifier.pkl")

# ======================
# Camera
# ======================
picam2 = Picamera2()
picam2.start()

# ======================
# Preprocess
# ======================
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))   # smaller = faster
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))    # HWC → CHW
    img = np.expand_dims(img, axis=0)     # add batch
    return img

# ======================
# Detection
# ======================
def detect(frame):
    img = preprocess(frame)
    outputs = session.run(None, {input_name: img})

    preds = outputs[0][0]

    boxes = []
    for pred in preds:
        conf = pred[4]

        if conf > 0.5:
            x1, y1, x2, y2 = pred[:4]

            # scale to original frame
            x1 = int(x1 * frame.shape[1] / 640)
            y1 = int(y1 * frame.shape[0] / 640)
            x2 = int(x2 * frame.shape[1] / 640)
            y2 = int(y2 * frame.shape[0] / 640)

            boxes.append((x1, y1, x2, y2, conf))

    return boxes

# ======================
# Main loop
# ======================
frame_count = 0

while True:
    frame = picam2.capture_array()

    # FIX: convert BGRA → BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Skip frames (performance boost)
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    boxes = detect(frame)

    if len(boxes) == 0:
        cv2.putText(frame, "NO BOTTLE", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        # pick center-most bottle
        h, w = frame.shape[:2]
        center_x = w // 2

        best_box = min(
            boxes,
            key=lambda b: abs(((b[0]+b[2])//2) - center_x)
        )

        x1, y1, x2, y2, conf = best_box

        # Draw box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # ======================
        # CLASSIFICATION
        # ======================
        bottle = frame[y1:y2, x1:x2]

        if bottle.size != 0:
            gray = cv2.cvtColor(bottle, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))
            features = gray.flatten().reshape(1, -1)

            prediction = clf.predict(features)[0]
            probs = clf.predict_proba(features)[0]
            confidence = max(probs) * 100

            if prediction == 0:
                label = f"GOOD ({confidence:.1f}%)"
                color = (0,255,0)
            else:
                label = f"DEFECTIVE ({confidence:.1f}%)"
                color = (0,0,255)

            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show frame
    cv2.imshow("Bottle Inspection System", frame)

    # ======================
    # EXIT VIDEO
    # ======================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.close()