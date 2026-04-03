import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time

session = ort.InferenceSession("yolov8n.onnx")
input_name = session.get_inputs()[0].name

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

print(f"Frame size: {w}x{h}")
print("Detections:")
print("-" * 50)

# Preprocess
img = cv2.resize(frame, (320, 320))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

# Run inference
outputs = session.run(None, {input_name: img})
preds = outputs[0][0]

for i, pred in enumerate(preds):
    conf = pred[4]
    if conf > 0.3:
        class_probs = pred[5:]
        class_id = np.argmax(class_probs)
        class_conf = class_probs[class_id]
        
        print(f"Detection {i}: Class={class_id}, Objectness={conf:.2f}, Class_conf={class_conf:.2f}")
        
        if class_id == 39:  # Bottle class
            x1, y1, x2, y2 = pred[:4]
            print(f"  Box in 320 space: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})")
            
            # Scale to frame
            x1_scaled = int(x1 * w / 320)
            y1_scaled = int(y1 * h / 320)
            x2_scaled = int(x2 * w / 320)
            y2_scaled = int(y2 * h / 320)
            print(f"  Box in frame space: ({x1_scaled},{y1_scaled}) to ({x2_scaled},{y2_scaled})")
            print(f"  Box size: {x2_scaled-x1_scaled} x {y2_scaled-y1_scaled}")
            print(f"  GOOD - This is a bottle detection!")
        else:
            print(f"  NOT a bottle (class {class_id})")
        
        print("-" * 30)

picam2.close()
print("\nDiagnostic complete. Press Enter to exit.")
input()