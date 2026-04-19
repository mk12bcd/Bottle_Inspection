import socket
import numpy as np
import cv2
import torch

# Load your trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp10/weights/best.pt')

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5000))
server.listen(1)

print("Waiting for Pi...")

conn, addr = server.accept()
print("Connected:", addr)

def recvall(conn, size):
    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

while True:
    try:
        size = int.from_bytes(conn.recv(4), 'big')
        img_data = recvall(conn, size)

        img = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # 🔥 RUN YOLO
        results = model(frame)

        # get prediction text
        labels = results.pandas().xyxy[0]['name'].tolist()

        if len(labels) == 0:
            msg = "no_object"
        else:
            msg = labels[0]

        print("Detected:", msg)

        conn.send(msg.encode())

    except Exception as e:
        print("Error:", e)
        break

conn.close()
server.close()