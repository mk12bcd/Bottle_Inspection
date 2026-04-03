import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time

# Load model
session = ort.InferenceSession("yolov8n.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Get input shape
input_shape = session.get_inputs()[0].shape
print(f"Model expects input shape: {input_shape}")

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    buffer_count=2
)
picam2.configure(config)
picam2.start()
time.sleep(1)

frame = picam2.capture_array()
h, w = frame.shape[:2]

# Preprocess - resize to model input size
img = cv2.resize(frame, (320, 320))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

# Run inference
outputs = session.run(None, {input_name: img})

print(f"\nOutput shape: {outputs[0].shape}")
print(f"Frame size: {w}x{h}")
print("=" * 50)

# YOLO output format: [1, 84, 8400] or [1, 8400, 84]
# Let's check the shape
output = outputs[0]
print(f"Output tensor shape: {output.shape}")

if len(output.shape) == 3:
    if output.shape[1] == 84:
        # Shape is [1, 84, 8400] - transpose needed
        output = output.transpose(0, 2, 1)
    
    # Now shape should be [1, num_detections, 84]
    detections = output[0]
    
    print(f"Number of detections: {len(detections)}")
    print("-" * 50)
    
    for i in range(min(10, len(detections))):
        detection = detections[i]
        conf = detection[4]
        
        if conf > 0.3:
            # Get class scores (positions 5-84)
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            print(f"Detection {i}:")
            print(f"  Objectness: {conf:.3f}")
            print(f"  Class ID: {class_id}")
            print(f"  Class Confidence: {class_conf:.3f}")
            
            if class_id == 39:
                x1, y1, x2, y2 = detection[:4]
                
                # Scale to original frame
                x1 = int(x1 * w / 320)
                y1 = int(y1 * h / 320)
                x2 = int(x2 * w / 320)
                y2 = int(y2 * h / 320)
                
                print(f"  Box: ({x1},{y1}) to ({x2},{y2})")
                print(f"  ✅ BOTTLE DETECTED!")
            else:
                print(f"  ❌ Not a bottle (class {class_id})")
            print("-" * 30)

picam2.close()
print("\nDiagnostic complete.")