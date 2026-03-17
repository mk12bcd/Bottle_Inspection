import cv2
import joblib
import numpy as np

# Load model
clf = joblib.load("bottle_classifier.pkl")

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (200, 200))
    features = gray_resized.flatten().reshape(1, -1)

    # Predict
    prediction = clf.predict(features)
    probs = clf.predict_proba(features)
    confidence = max(probs[0]) * 100

    # Clean label
    label = prediction[0]
    if "Good" in label:
        display_text = f"GOOD BOTTLE ({confidence:.1f}%)"
        color = (0, 255, 0)
    else:
        display_text = f"{label.upper()} ({confidence:.1f}%)"
        color = (0, 0, 255)

    # Display on frame
    cv2.putText(frame, display_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show camera preview
    cv2.imshow("Camera", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()