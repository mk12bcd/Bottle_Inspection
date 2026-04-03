import cv2
import numpy as np
import ncnn
from picamera2 import Picamera2
import time

# Load NCNN model
net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.load_param("yolov8n_ncnn_model/model.ncnn.param")
net.load_model("yolov8n_ncnn_model/model.ncnn.bin")

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    buffer_count=2
)
picam2.configure(config)
picam2.start()
time.sleep(1)

print("Taking a picture...")
frame = picam2.capture_array()
h, w = frame.shape[:2]

# Preprocess
img = cv2.resize(frame, (320, 320))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create NCNN mat
mat_in = ncnn.Mat.from_pixels(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 320, 320)
mat_in.substract_mean_normalize([0, 0, 0], [1/255, 1/255, 1/255])

# Run inference
ex = net.create_extractor()
ex.input("in0", mat_in)

ret, mat_out = ex.extract("out0")
data = np.array(mat_out)

print(f"\nOutput shape: {data.shape}")
print(f"Number of detections: {data.shape[0]}")
print("=" * 60)

# Show all detections
bottle_count = 0
for i in range(data.shape[0]):
    obj_conf = data[i][4]
    
    if obj_conf > 0.3:
        class_scores = data[i][5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        
        print(f"\nDetection {i}:")
        print(f"  Objectness: {obj_conf:.3f}")
        print(f"  Class ID: {class_id}")
        print(f"  Class Confidence: {class_conf:.3f}")
        
        if class_id == 39:
            bottle_count += 1
            x1 = data[i][0]
            y1 = data[i][1]
            x2 = data[i][2]
            y2 = data[i][3]
            
            print(f"  ✅ BOTTLE DETECTED!")
            print(f"  Raw box coords: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
            
            # Scale to frame
            x1_scaled = int(x1 * w / 320)
            y1_scaled = int(y1 * h / 320)
            x2_scaled = int(x2 * w / 320)
            y2_scaled = int(y2 * h / 320)
            
            print(f"  Scaled box: ({x1_scaled},{y1_scaled}) to ({x2_scaled},{y2_scaled})")
            
            # Draw on frame
            cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
            cv2.putText(frame, f"BOTTLE {obj_conf:.2f}", (x1_scaled, y1_scaled - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f"\n{'='*60}")
print(f"Total bottles detected: {bottle_count}")

if bottle_count == 0:
    print("\n⚠️ NO BOTTLES DETECTED")
    max_conf = np.max(data[:, 4])
    print(f"Highest object confidence: {max_conf:.3f}")
    
    # Show unique classes detected
    unique_classes = set()
    for i in range(data.shape[0]):
        if data[i][4] > 0.2:
            class_id = np.argmax(data[i][5:])
            unique_classes.add(class_id)
    print(f"Classes detected (above 0.2): {sorted(unique_classes)}")

# Save image
cv2.imwrite("debug_output.jpg", frame)
print("\nImage saved as debug_output.jpg")

picam2.close()