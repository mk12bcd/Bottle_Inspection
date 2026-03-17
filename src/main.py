# main.py
from picamera2 import Picamera2
import cv2
import joblib
import numpy as np
import os

# ======================
# 1️⃣ Load trained classifier
# ======================
MODEL_PATH = "bottle_classifier.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Make sure train_classifier.py has run successfully.")

clf = joblib.load(MODEL_PATH)

# ======================
# 2️⃣ Initialize Picamera2
# ======================
picam2 = Picamera2()
picam2.start()

# ======================
# 3️⃣ Real-time camera loop
# ======================
print("Starting live bottle classification. Press 'q' to quit.")

while True:
    # Capture frame
    frame = picam2.capture_array()  # returns a numpy array (BGR)

    # Convert to grayscale & resize for classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    cv2.putText(frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show live preview
    cv2.imshow("Bottle Classifier", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================
# 4️⃣ Cleanup
# ======================
cv2.destroyAllWindows()
picam2.close()
print("Camera closed. Exiting.")