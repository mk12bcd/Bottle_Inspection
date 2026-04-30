from picamera2 import Picamera2
import socket
import cv2
import RPi.GPIO as GPIO
import signal
import sys

RELAY_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

RELAY_ON = 0
RELAY_OFF = 1
GPIO.output(RELAY_PIN, RELAY_OFF)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

PC_IP = "172.18.80.1"
PORT = 5000

s = socket.socket()
s.connect((PC_IP, PORT))
s.settimeout(1.0)

good_count = 0
no_cap_count = 0
no_label_count = 0

latest_id = "-"
latest_class = "waiting"

running = True

def stop(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

def send_frame():
    frame = picam2.capture_array()
    _, buffer = cv2.imencode(".jpg", frame)
    data = buffer.tobytes()
    size = str(len(data)).ljust(16).encode()
    s.sendall(size)
    s.sendall(data)

while running:

    try:
        cmd = s.recv(16).decode().strip()
    except:
        cmd = None

    if cmd == "CAPTURE":
        send_frame()

    elif cmd and cmd.startswith("ID:"):
        try:
            parts = cmd.split("|")
            latest_id = parts[0].split(":")[1]
            latest_class = parts[1]
            good = int(parts[2].split(":")[1])
            defects = int(parts[3].split(":")[1])
        except:
            continue

        if latest_class == "good":
            good_count += 1
            GPIO.output(RELAY_PIN, RELAY_OFF)
        elif latest_class == "no_cap":
            no_cap_count += 1
            GPIO.output(RELAY_PIN, RELAY_ON)
        elif latest_class == "no_label":
            no_label_count += 1
            GPIO.output(RELAY_PIN, RELAY_ON)

    frame = picam2.capture_array()
    h, w, _ = frame.shape

    # TITLE (UNCHANGED STYLE)
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, "MYK AUTOMATION", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # COUNTERS (BOTTOM LEFT)
    cv2.putText(frame, f"Good: {good_count}", (20, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"No Cap: {no_cap_count}", (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"No Label: {no_label_count}", (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # CURRENT BOTTLE (BOTTOM RIGHT, BLACK TEXT WITH WHITE BOX)
    status = f"Bottle {latest_id}: {latest_class}"

    (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    x = w - tw - 20
    y = h - 20

    cv2.rectangle(frame,
                  (x - 10, y - th - 10),
                  (x + tw + 10, y + 10),
                  (255, 255, 255), -1)

    cv2.putText(frame, status,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2)

    cv2.imshow("Bottle Inspection", frame)

    if cv2.waitKey(1) == 27:
        running = False

    if cv2.getWindowProperty("Bottle Inspection", cv2.WND_PROP_VISIBLE) < 1:
        running = False

GPIO.output(RELAY_PIN, RELAY_OFF)
GPIO.cleanup()
picam2.stop()
s.close()
cv2.destroyAllWindows()
sys.exit(0)