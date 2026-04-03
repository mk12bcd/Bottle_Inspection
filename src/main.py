import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import joblib
import time
from collections import deque

# ======================
# Load ONNX model (320x320)
# ======================
session = ort.InferenceSession("yolov8n.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

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

# ======================
# Stability for classification (0.7 second delay)
# ======================
predictions_buffer = []
confidence_buffer = []
detection_start_time = None
current_bottle_id = None
last_classified_time = 0
CLASSIFICATION_DELAY = 0.7  # 0.7 seconds

# Cooldown to prevent double counting
last_counted_bottle = None
COUNTING_COOLDOWN = 1.5  # seconds

# ======================
# Preprocess for ONNX
# ======================
def preprocess(frame):
    h, w = frame.shape[:2]
    scale = 320 / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create square canvas (320x320)
    canvas = np.full((320, 320, 3), 114, dtype=np.uint8)
    x_offset = (320 - new_w) // 2
    y_offset = (320 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Normalize and convert
    img = canvas.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img, (x_offset, y_offset, scale)

# ======================
# Detection with ONNX
# ======================
def detect(frame):
    img, (x_offset, y_offset, scale) = preprocess(frame)
    outputs = session.run(None, {input_name: img})
    
    preds = outputs[0][0]
    boxes = []
    
    for pred in preds:
        conf = pred[4]
        if conf > 0.5:
            x1, y1, x2, y2 = pred[:4]
            
            # Remove padding offset
            x1 = (x1 - x_offset) / scale
            y1 = (y1 - y_offset) / scale
            x2 = (x2 - x_offset) / scale
            y2 = (y2 - y_offset) / scale
            
            # Clip to frame boundaries
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))
            
            # Filter out tiny boxes
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                boxes.append((x1, y1, x2, y2, conf))
    
    return boxes

print("=" * 50)
print("MKY Automation - Bottle Inspection System")
print("=" * 50)
print("Press 'q' to quit")
print("Press 'r' to reset counters")
print("=" * 50)

# ======================
# Main loop
# ======================
while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape
    
    # ======================
    # Detect bottles
    # ======================
    boxes = detect(frame)
    
    # ======================
    # Find bottle closest to center
    # ======================
    center_x = w // 2
    best_box = None
    min_distance = float("inf")
    
    for box in boxes:
        x1, y1, x2, y2, conf = box
        box_center_x = (x1 + x2) // 2
        distance = abs(box_center_x - center_x)
        
        if distance < min_distance:
            min_distance = distance
            best_box = box
    
    # ======================
    # Create unique bottle ID
    # ======================
    bottle_id = None
    if best_box:
        x1, y1, x2, y2, conf = best_box
        bottle_id = f"{x1//30}_{y1//30}"
    
    # ======================
    # Handle detection and classification with 0.7s delay
    # ======================
    current_time = time.time()
    
    if best_box and bottle_id:
        x1, y1, x2, y2, conf = best_box
        
        # Draw bounding box (always show)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"BOTTLE", (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # New bottle detected
        if current_bottle_id != bottle_id:
            # Reset for new bottle
            current_bottle_id = bottle_id
            detection_start_time = current_time
            predictions_buffer = []
            confidence_buffer = []
        
        # Show "DETECTING..." during the 0.7 second delay
        if detection_start_time and (current_time - detection_start_time) < CLASSIFICATION_DELAY:
            remaining = int(CLASSIFICATION_DELAY - (current_time - detection_start_time))
            cv2.putText(frame, f"ANALYZING... ({remaining}s)", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # After delay, classify
        elif detection_start_time and (current_time - detection_start_time) >= CLASSIFICATION_DELAY:
            # Extract bottle ROI
            bottle_roi = frame[y1:y2, x1:x2]
            
            if bottle_roi.size != 0 and (current_time - last_classified_time) > 0.3:
                # Preprocess for classifier
                gray = cv2.cvtColor(bottle_roi, cv2.COLOR_RGB2GRAY)
                gray = cv2.resize(gray, (200, 200))
                features = gray.flatten().reshape(1, -1)
                
                # Predict
                prediction = clf.predict(features)[0]
                probs = clf.predict_proba(features)[0]
                confidence = max(probs) * 100
                
                # Add to buffers for stability
                predictions_buffer.append(prediction)
                confidence_buffer.append(confidence)
                
                if len(predictions_buffer) > 3:
                    predictions_buffer.pop(0)
                    confidence_buffer.pop(0)
                
                last_classified_time = current_time
            
            # Get stable prediction
            if predictions_buffer:
                final_pred = max(set(predictions_buffer), key=predictions_buffer.count)
                final_conf = sum(confidence_buffer) / len(confidence_buffer)
                
                # Set color and label
                if final_pred == 0:
                    label = f"GOOD ({final_conf:.1f}%)"
                    color = (0, 255, 0)
                else:
                    label = f"{label_map[final_pred]} ({final_conf:.1f}%)"
                    color = (0, 0, 255)
                
                # Draw classification on bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Counting with cooldown
                if last_counted_bottle != bottle_id:
                    total_count += 1
                    if final_pred == 0:
                        good_count += 1
                    else:
                        defective_count += 1
                    last_counted_bottle = bottle_id
    else:
        # No bottle detected - reset
        current_bottle_id = None
        detection_start_time = None
        predictions_buffer = []
        confidence_buffer = []
        cv2.putText(frame, "NO BOTTLE", (w//2 - 60, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # ======================
    # UI Display
    # ======================
    
    # Title: MKY Automation (top center)
    cv2.putText(frame, "MKY AUTOMATION", (w//2 - 100, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Counters at bottom right
    cv2.putText(frame, f"TOTAL: {total_count}", (w - 150, h - 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"GOOD: {good_count}", (w - 150, h - 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"DEFECTIVE: {defective_count}", (w - 150, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Center marker (crosshair)
    cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 255, 0), 1)
    cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 255, 0), 1)
    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 0), -1)
    
    # ======================
    # Show frame
    # ======================
    cv2.imshow("MKY Automation - Bottle Inspection", frame)
    
    # ======================
    # Keyboard controls
    # ======================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        good_count = 0
        defective_count = 0
        total_count = 0
        last_counted_bottle = None
        print("Counters reset!")

# ======================
# Cleanup
# ======================
cv2.destroyAllWindows()
picam2.close()

print("\n" + "=" * 50)
print("MKY AUTOMATION - FINAL REPORT")
print("=" * 50)
print(f"Total Bottles: {total_count}")
print(f"Good Bottles: {good_count}")
print(f"Defective Bottles: {defective_count}")
print("=" * 50)