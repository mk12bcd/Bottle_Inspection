import cv2
import numpy as np
import ncnn
from picamera2 import Picamera2
import joblib
import time

# ======================
# Load NCNN model
# ======================
net = ncnn.Net()
net.opt.use_vulkan_compute = False  # Use CPU only
net.load_param("yolov8n_ncnn_model/model.ncnn.param")
net.load_model("yolov8n_ncnn_model/model.ncnn.bin")

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
# Detection with NCNN
# ======================
def detect(frame):
    h, w = frame.shape[:2]
    
    # Preprocess
    img = cv2.resize(frame, (320, 320))
    
    # Create NCNN mat
    mat_in = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_RGB, 320, 320)
    mat_in.substract_mean_normalize([0, 0, 0], [1/255, 1/255, 1/255])
    
    # Run inference
    ex = net.create_extractor()
    ex.input("images", mat_in)
    
    ret, mat_out = ex.extract("output")
    
    # Parse output
    boxes = []
    data = np.array(mat_out)
    
    # data shape: [num_detections, 84]
    for i in range(data.shape[0]):
        obj_conf = data[i][4]
        
        if obj_conf > 0.4:
            class_scores = data[i][5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            if class_id == 39 and class_conf > 0.4:
                x1 = int(data[i][0] * w / 320)
                y1 = int(data[i][1] * h / 320)
                x2 = int(data[i][2] * w / 320)
                y2 = int(data[i][3] * h / 320)
                
                # Ensure coordinates are valid
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                if (x2 - x1) > 30 and (y2 - y1) > 30:
                    boxes.append((x1, y1, x2, y2, obj_conf))
    
    return boxes

print("=" * 50)
print("MKY Automation - Bottle Inspection System (NCNN)")
print("=" * 50)
print("Press 'q' to quit")
print("Press 'r' to reset counters")
print("=" * 50)

# ======================
# Main loop
# ======================
frame_count = 0
current_bottle_id = None
detection_start_time = None
predictions_buffer = []
last_counted_bottle = None
CLASSIFICATION_DELAY = 0.7

while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape
    
    # Run detection every frame
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
        
        # Show analyzing during delay
        if detection_start_time and (current_time - detection_start_time) < CLASSIFICATION_DELAY:
            remaining = int(CLASSIFICATION_DELAY - (current_time - detection_start_time))
            cv2.putText(frame, f"ANALYZING... ({remaining}s)", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Classify after delay
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
                    
                    if last_counted_bottle != bottle_id:
                        total_count += 1
                        if final_pred == 0:
                            good_count += 1
                        else:
                            defective_count += 1
                        last_counted_bottle = bottle_id
    else:
        current_bottle_id = None
        detection_start_time = None
        predictions_buffer = []
        cv2.putText(frame, "NO BOTTLE", (w//2 - 60, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # UI Display
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
        last_counted_bottle = None
        print("Counters reset!")

cv2.destroyAllWindows()
picam2.close()

print("\n" + "=" * 50)
print("MKY AUTOMATION - FINAL REPORT")
print("=" * 50)
print(f"Total Bottles: {total_count}")
print(f"Good Bottles: {good_count}")
print(f"Defective Bottles: {defective_count}")
print("=" * 50)