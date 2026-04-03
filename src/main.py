import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import joblib
import time
from collections import deque

# ======================
# Load ONNX model
# ======================
session = ort.InferenceSession("yolov8n.onnx", providers=['CPUExecutionProvider'])  # Force CPU
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # Get expected input shape

# ======================
# Load classifier
# ======================
clf = joblib.load("bottle_classifier.pkl")

# ======================
# Camera configuration for less lag
# ======================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},  # Smaller resolution, RGB format
    buffer_count=2  # Reduce buffer count for less latency
)
picam2.configure(config)
picam2.start()
time.sleep(0.5)  # Allow camera to warm up

# ======================
# Statistics for counting
# ======================
good_count = 0
defective_count = 0
total_count = 0
fps_buffer = deque(maxlen=30)
last_detection_time = {}
DETECTION_COOLDOWN = 1.0  # seconds between counting same bottle

# ======================
# Preprocess - FIXED for 640x640
# ======================
def preprocess(frame):
    # Resize maintaining aspect ratio with padding
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create square canvas (640x640)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)  # Gray padding
    x_offset = (640 - new_w) // 2
    y_offset = (640 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Normalize and convert
    img = canvas.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img, (x_offset, y_offset, scale)

# ======================
# Detection with proper scaling
# ======================
def detect(frame):
    img, (x_offset, y_offset, scale) = preprocess(frame)
    outputs = session.run(None, {input_name: img})
    
    preds = outputs[0][0]
    boxes = []
    
    for pred in preds:
        conf = pred[4]
        if conf > 0.5:  # Confidence threshold
            # Get box coordinates in 640x640 space
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
            
            # Filter out tiny boxes (likely false positives)
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                boxes.append((x1, y1, x2, y2, conf))
    
    return boxes

# ======================
# Calculate FPS
# ======================
def update_fps():
    fps_buffer.append(time.time())
    if len(fps_buffer) > 1:
        fps = len(fps_buffer) / (fps_buffer[-1] - fps_buffer[0])
        return fps
    return 0

# ======================
# Main loop with performance optimizations
# ======================
frame_count = 0
process_every_n_frames = 2  # Process every frame (reduce lag)
last_frame_time = time.time()

print("Starting bottle inspection system...")
print("Press 'q' to quit")
print("Press 'r' to reset counters")
print("Press 'd' to toggle detection display")

show_detections = True

while True:
    # Capture frame
    frame = picam2.capture_array()
    
    # Frame is already RGB from camera config
    current_time = time.time()
    
    # Process at reduced frequency for better performance
    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        # Still show frame but skip detection
        display_frame = frame.copy()
    else:
        # Run detection
        boxes = detect(frame)
        display_frame = frame.copy()
        
        # Bottle counting and classification
        if len(boxes) == 0:
            cv2.putText(display_frame, "NO BOTTLE DETECTED", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Find center-most bottle
            h, w = display_frame.shape[:2]
            center_x = w // 2
            
            best_box = min(boxes, key=lambda b: abs(((b[0] + b[2]) // 2) - center_x))
            x1, y1, x2, y2, conf = best_box
            
            # Create unique ID for this bottle position
            bottle_id = f"{x1//50}_{y1//50}"
            
            # Check if we should count this bottle
            if bottle_id not in last_detection_time or \
               (current_time - last_detection_time[bottle_id]) > DETECTION_COOLDOWN:
                
                # Extract and classify bottle ROI
                bottle = frame[y1:y2, x1:x2]
                
                if bottle.size != 0:
                    # Improved feature extraction
                    gray = cv2.cvtColor(bottle, cv2.COLOR_RGB2GRAY)
                    gray = cv2.resize(gray, (200, 200))
                    
                    # Use HOG-like features or just flatten
                    features = gray.flatten().reshape(1, -1)
                    
                    prediction = clf.predict(features)[0]
                    probs = clf.predict_proba(features)[0]
                    confidence = max(probs) * 100
                    
                    # Update counters
                    total_count += 1
                    if prediction == 0:
                        good_count += 1
                        label = f"GOOD ({confidence:.1f}%)"
                        color = (0, 255, 0)
                    else:
                        defective_count += 1
                        label = f"DEFECTIVE ({confidence:.1f}%)"
                        color = (0, 0, 255)
                    
                    last_detection_time[bottle_id] = current_time
                    
                    # Draw classification result
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Conf: {conf:.2f}", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display statistics
        fps = update_fps()
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Total: {total_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Good: {good_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Defective: {defective_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show processing time
        process_time = (time.time() - current_time) * 1000
        cv2.putText(display_frame, f"Process: {process_time:.0f}ms", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show frame
    cv2.imshow("Bottle Inspection System", display_frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        good_count = 0
        defective_count = 0
        total_count = 0
        last_detection_time.clear()
        print("Counters reset")
    elif key == ord('d'):
        show_detections = not show_detections
    elif key == ord('+') and process_every_n_frames > 1:
        process_every_n_frames -= 1
        print(f"Processing every {process_every_n_frames} frames")
    elif key == ord('-'):
        process_every_n_frames += 1
        print(f"Processing every {process_every_n_frames} frames")

# Cleanup
cv2.destroyAllWindows()
picam2.close()
print(f"\nFinal Statistics:")
print(f"Total bottles: {total_count}")
print(f"Good: {good_count}")
print(f"Defective: {defective_count}")