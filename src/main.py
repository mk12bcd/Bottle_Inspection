import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

# Load ONNX model
session = ort.InferenceSession("yolov8n.onnx")

# Camera
picam2 = Picamera2()
picam2.start()

input_name = session.get_inputs()[0].name

def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    return img

def detect(frame):
    img = preprocess(frame)

    outputs = session.run(None, {input_name: img})

    predictions = outputs[0][0]

    boxes = []
    for pred in predictions:
        conf = pred[4]

        if conf > 0.5:
            x, y, w, h = pred[:4]

            x1 = int((x - w/2) * frame.shape[1] / 640)
            y1 = int((y - h/2) * frame.shape[0] / 640)
            x2 = int((x + w/2) * frame.shape[1] / 640)
            y2 = int((y + h/2) * frame.shape[0] / 640)

            boxes.append((x1, y1, x2, y2, conf))

    return boxes

while True:
    frame = picam2.capture_array()

    boxes = detect(frame)

    if len(boxes) == 0:
        cv2.putText(frame, "NO BOTTLE", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        # choose center-most box
        h, w = frame.shape[:2]
        center_x = w // 2

        best_box = min(boxes, key=lambda b: abs(((b[0]+b[2])//2) - center_x))

        x1, y1, x2, y2, conf = best_box

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"BOTTLE {conf:.2f}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("ONNX Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()