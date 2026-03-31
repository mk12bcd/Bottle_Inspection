from picamera2 import Picamera2
import cv2
import joblib
import numpy as np

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

# ======================
# Bottle Detection Function
# ======================
def detect_bottle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    if w * h < 2000:  # filter noise
        return None

    return (x, y, x+w, y+h)

print("Running... Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape

    # Smooth frame
    frame = cv2.GaussianBlur(frame, (3,3), 0)

    # Title
    cv2.putText(frame, "Bottle Inspection System", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Detect bottle
    box = detect_bottle(frame)

    bottle_detected = False

    if box is not None:
        x1, y1, x2, y2 = box
        bottle_detected = True

        bottle = frame[y1:y2, x1:x2]

        if bottle.size != 0:
            gray = cv2.cvtColor(bottle, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))
            features = gray.flatten().reshape(1, -1)

            prediction = clf.predict(features)[0]
            probs = clf.predict_proba(features)[0]
            confidence = max(probs) * 100

            predictions_buffer.append(prediction)
            confidence_buffer.append(confidence)

            if len(predictions_buffer) > 3:
                predictions_buffer.pop(0)
                confidence_buffer.pop(0)

            stable_count += 1
            cooldown -= 1

            # Detecting state
            if stable_count <= 1:
                cv2.putText(frame, "DETECTING...", (30,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

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

                if not last_counted and cooldown <= 0:
                    if final_pred == 0:
                        good_count += 1
                    else:
                        defective_count += 1

                    last_counted = True
                    cooldown = 10

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    else:
        stable_count = 0
        predictions_buffer.clear()
        confidence_buffer.clear()
        last_counted = False

        cv2.putText(frame, "NO BOTTLE", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Counters
    cv2.putText(frame, f"GOOD: {good_count}", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"DEFECTIVE: {defective_count}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Bottle Inspection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()