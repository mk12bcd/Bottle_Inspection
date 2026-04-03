import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import joblib
import time

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
# COCO class names (for filtering)
# ======================
# Class 39 is "bottle" in COCO dataset
BOTTLE_CLASS_ID = 39

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
CLASSIFICATION_DELAY = 0.7

last_counted_bottle = None

# ======================
# Preprocess for ONNX
# ======================
def preprocess(frame):
    h, w = frame.shape[:2]
    
    # Resize directly to 320x320 (no padding, simpler)
    img = cv2.resize(frame, (320, 320))
    
    # Normalize and convert
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img, (w, h)  # Return original dimensions for scaling

# ======================
# Detection with ONNX (filter for bottle class)
# ======================
def detect(frame):
    img, (orig_w, orig_h) = preprocess(frame)
    outputs = session.run(None, {input_name: img})
    
    # YOLO output format: [batch, num_detections, 85]
    # 85 = [x1, y1, x2, y2, objectness, class_probs...]
    preds = outputs[0][0]
    
    boxes = []
    
    for pred in preds:
        # Get confidence
        conf = pred[4]
        
        if conf > 0.5:
            # Get class ID (argmax of class probabilities)
            class_probs = pred[5:]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            # Combined confidence
            total_conf = conf * class_conf
            
            # Only detect bottles (class 39)
            if class_id == BOTTLE_CLASS_ID and total_conf > 0.4:
                # Get coordinates (these are in 320x320 space)
                x1 = pred[0]
                y1 = pred[1]
                x2 = pred[2]
                y2 = pred[3]
                
                # Scale to original frame size
                x1 = int(x1 * orig_w / 320)
                y1 = int(y1 * orig_h / 320)
                x2 = int(x2 * orig_w / 320)
                y2 = int(y2 * orig_h / 320)
                
                # Ensure coordinates are within frame
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                # Filter out tiny boxes
                if (x2 - x1) > 30 and (y2 - y1) > 30:
                    boxes.append((x1, y1, x2, y2, total_conf))
                    print(f"Bottle detected at: ({x1},{y1}) to ({x2},{y2}) with confidence: {total_conf:.2f}")
    
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
        
        # Draw bounding box (yellow while detecting)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # New bottle detected
        if current_bottle_id != bottle_id:
            # Reset for new bottle
            current_bottle_id = bottle_id
            detection_start_time = current_time
            predictions_buffer = []
            confidence_buffer = []
        
        # Show "ANALYZING..." during the 0.7 second delay
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