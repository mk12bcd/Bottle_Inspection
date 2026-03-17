# main.py
# main.py
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import joblib
import numpy as np
import time
import os

# ======================
# 1️⃣ Load trained classifier
# ======================
MODEL_PATH = "bottle_classifier.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Make sure train_classifier.py has run successfully.")

clf = joblib.load(MODEL_PATH)

# ======================
# 2️⃣ Initialize PiCamera
# ======================
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
raw_capture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)  # allow camera to warm up

# ======================
# 3️⃣ Real-time capture loop
# ======================
print("Starting live bottle classification. Press 'q' to quit.")

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    image = frame.array

    # Convert to grayscale & resize for classifier
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (200, 200))
    features = gray_resized.flatten().reshape(1, -1)

    # Predict class
    prediction = clf.predict(features)[0]
    probs = clf.predict_proba(features)[0]
    confidence = max(probs) * 100

    # Color-coded label
    if "Good" in prediction:
        text = f"GOOD BOTTLE ({confidence:.1f}%)"
        color = (0, 255, 0)  # Green
    else:
        text = f"{prediction.upper()} ({confidence:.1f}%)"
        color = (0, 0, 255)  # Red

    # Display label on frame
    cv2.putText(image, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show live preview
    cv2.imshow("Bottle Classifier", image)

    key = cv2.waitKey(1) & 0xFF
    raw_capture.truncate(0)  # clear stream for next frame

    if key == ord('q'):
        break

# ======================
# 4️⃣ Cleanup
# ======================
cv2.destroyAllWindows()
camera.close()
print("Camera closed. Exiting.")