from picamera2 import Picamera2
import socket
import cv2
import RPi.GPIO as GPIO
import signal
import sys
import time

# ================= RELAY =================
RELAY_PIN = 18  # GPIO18 (Pin 12)

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

# ACTIVE LOW RELAY
RELAY_ON = 0
RELAY_OFF = 1

GPIO.output(RELAY_PIN, RELAY_OFF)

# ================= CAMERA =================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# ================= SOCKET =================
PC_IP = "192.168.1.7"   # <-- CHANGE IF NEEDED
PORT = 5000

s = socket.socket()
s.connect((PC_IP, PORT))
s.settimeout(1.0)

print("[PI] Connected to PC")

# ================= STATE =================
latest_id = "-"
latest_class = "waiting"

good_count = 0
no_cap_count = 0
no_label_count = 0

running = True


# ================= SAFE EXIT =================
def stop(sig, frame):
    global running
    print("\n[PI] Stopping safely...")
    running = False


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)


# ================= SEND FRAME =================
def send_frame():
    frame = picam2.capture_array()
    _, buffer = cv2.imencode(".jpg", frame)

    data = buffer.tobytes()
    size = str(len(data)).ljust(16).encode()

    s.sendall(size)
    s.sendall(data)


# ================= MAIN LOOP =================
while running:

    # -------- RECEIVE COMMAND --------
    try:
        cmd = s.recv(16).decode().strip()
    except:
        cmd = None

    if not running:
        break

    # -------- SEND FRAME --------
    if cmd == "CAPTURE":
        send_frame()

    # -------- RECEIVE RESULT --------
    elif cmd and cmd.startswith("ID:"):
        try:
            parts = cmd.split("|")
            latest_id = parts[0].split(":")[1]
            latest_class = parts[1].split(":")[1]
        except:
            continue

        print(f"[PI] Bottle {latest_id} → {latest_class}")

        # ================= RELAY LOGIC =================
        if latest_class == "good":
            good_count += 1
            GPIO.output(RELAY_PIN, RELAY_OFF)

        elif latest_class == "no_cap":
            no_cap_count += 1
            GPIO.output(RELAY_PIN, RELAY_ON)

        elif latest_class == "no_label":
            no_label_count += 1
            GPIO.output(RELAY_PIN, RELAY_ON)

    # ================= UI FRAME =================
    frame = picam2.capture_array()
    h, w, _ = frame.shape

    # TITLE
    cv2.putText(frame,
                "MYK AUTOMATION",
                (w//2 - 180, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3)

    # COUNTERS (BOTTOM LEFT)
    cv2.putText(frame,
                f"Good: {good_count}",
                (20, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                f"No Cap: {no_cap_count}",
                (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)

    cv2.putText(frame,
                f"No Label: {no_label_count}",
                (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2)

    # STATUS (BOTTOM RIGHT)
    status = f"ID {latest_id} | {latest_class}"

    color = (0, 255, 0) if latest_class == "good" else (0, 0, 255)

    (tw, th), _ = cv2.getTextSize(status,
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.8,
                                  2)

    cv2.putText(frame,
                status,
                (w - tw - 20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2)

    cv2.imshow("Bottle Inspection", frame)

    # EXIT CONDITIONS
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        running = False

    if cv2.getWindowProperty("Bottle Inspection", cv2.WND_PROP_VISIBLE) < 1:
        running = False


# ================= CLEANUP =================
print("[PI] Cleaning up...")

GPIO.output(RELAY_PIN, RELAY_OFF)
GPIO.cleanup()

picam2.stop()
s.close()
cv2.destroyAllWindows()

sys.exit(0)