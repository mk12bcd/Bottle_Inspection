from picamera2 import Picamera2
import socket
import cv2
import RPi.GPIO as GPIO
import signal
import sys

# ================= RELAY =================
RELAY_PIN = 18  # GPIO18 (Pin 12)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, 0)

# ================= CAMERA =================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# ================= SOCKET =================
s = socket.socket()
s.connect(("192.168.100.55", 5000))  # PC IP
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


# ================= LOOP =================
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

        # -------- COUNTERS --------
        if latest_class == "good":
            good_count += 1
            GPIO.output(RELAY_PIN, 0)

        elif latest_class == "no_cap":
            no_cap_count += 1
            GPIO.output(RELAY_PIN, 1)

        elif latest_class == "no_label":
            no_label_count += 1
            GPIO.output(RELAY_PIN, 1)

    # ================= UI =================
    frame = picam2.capture_array()
    h, w, _ = frame.shape

    # TOP TITLE
    cv2.putText(frame,
                "MYK AUTOMATION",
                (w//2 - 180, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3)

    # BOTTOM LEFT COUNTERS
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

    # BOTTOM RIGHT STATUS
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

    # EXIT CONTROL
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        running = False

    if cv2.getWindowProperty("Bottle Inspection", cv2.WND_PROP_VISIBLE) < 1:
        running = False

# ================= CLEANUP =================
print("[PI] Cleaning up...")

GPIO.output(RELAY_PIN, 0)
GPIO.cleanup()

picam2.stop()
s.close()
cv2.destroyAllWindows()

sys.exit(0)