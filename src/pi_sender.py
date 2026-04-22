from picamera2 import Picamera2
import socket
import cv2
import RPi.GPIO as GPIO
import time

# ---------------- GPIO SETUP ----------------
RELAY_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)  # default = GOOD state

def set_relay(state):
    GPIO.output(RELAY_PIN, GPIO.HIGH if state == 1 else GPIO.LOW)

# ---------------- CAMERA ----------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# ---------------- SOCKET ----------------
s = socket.socket()
s.connect(("192.168.100.11", 5000))

print("[PI] Connected")

latest_result = "waiting"

good = 0
defect = 0

while True:

    # ---------------- RECEIVE FROM PC ----------------
    try:
        data = s.recv(1024).decode().strip()

        if data.startswith("RESULT"):

            result = data.split(":")[1]
            latest_result = result

            # ---------------- RELAY LOGIC ----------------
            if result == "good":
                set_relay(0)   # GOOD → relay OFF
                good += 1

            elif result in ["no_cap", "no_label"]:
                set_relay(1)   # DEFECT → relay ON
                defect += 1

            else:
                set_relay(0)   # safe fallback

    except:
        pass

    # ---------------- DISPLAY ----------------
    frame = picam2.capture_array()
    h, w, _ = frame.shape

    cv2.putText(frame, "MYK AUTOMATION", (w//2 - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

    cv2.putText(frame, f"GOOD: {good}", (20, h-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"DEFECT: {defect}", (20, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    state_text = "GOOD (Relay OFF)" if latest_result == "good" else "DEFECT (Relay ON)"

    cv2.putText(frame, state_text, (w-350, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0,255,0) if latest_result=="good" else (0,0,255), 2)

    cv2.imshow("Pi Control System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------- CLEANUP ----------------
GPIO.output(RELAY_PIN, GPIO.LOW)
GPIO.cleanup()
s.close()
cv2.destroyAllWindows()