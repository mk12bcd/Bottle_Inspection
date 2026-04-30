from picamera2 import Picamera2
import socket
import cv2
import RPi.GPIO as GPIO
import signal
import sys

RELAY_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

GPIO.output(RELAY_PIN, GPIO.HIGH)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

s = socket.socket()
s.connect(("192.168.1.7", 5000))

buffer = ""

good = 0
no_cap = 0
no_label = 0

current_id = -1
current_class = "waiting"

def read_line(sock):
    global buffer
    data = sock.recv(4096).decode()
    if data:
        buffer += data

    if "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        return line.strip()

    return None

def stop(sig, frame):
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    GPIO.cleanup()
    picam2.stop()
    s.close()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)

while True:

    frame = picam2.capture_array()
    _, buf = cv2.imencode(".jpg", frame)
    data = buf.tobytes()

    size = str(len(data)).ljust(16).encode()
    s.sendall(size)
    s.sendall(data)

    msg = read_line(s)

    if msg and msg.startswith("ID:"):

        parts = msg.split("|")
        new_id = int(parts[0].split(":")[1])
        new_class = parts[1]

        if new_id != current_id:
            current_id = new_id
            current_class = new_class

            if new_class == "Good":
                good += 1
                GPIO.output(RELAY_PIN, GPIO.HIGH)

            elif new_class == "No_cap":
                no_cap += 1
                GPIO.output(RELAY_PIN, GPIO.LOW)

            elif new_class == "No_label":
                no_label += 1
                GPIO.output(RELAY_PIN, GPIO.LOW)

    h, w, _ = frame.shape

    cv2.putText(frame, "MYK AUTOMATION", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(frame, (w-320, h-70), (w-10, h-10), (255, 255, 255), -1)

    cv2.putText(frame, f"Current: {current_id} | {current_class}",
                (w-310, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame, f"Good: {good}", (20, h-90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"No Cap: {no_cap}", (20, h-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"No Label: {no_label}", (20, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Bottle Inspection", frame)

    if cv2.waitKey(1) == 27:
        break

GPIO.output(RELAY_PIN, GPIO.HIGH)
GPIO.cleanup()
picam2.stop()
s.close()
cv2.destroyAllWindows()