from picamera2 import Picamera2
import socket
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

s = socket.socket()
s.connect(("192.168.100.55", 5000))

print("[PI] Connected")

good = 0
no_cap = 0
no_label = 0

latest_id = "-"
latest_result = "Waiting"


def send_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    data = buffer.tobytes()

    size = str(len(data)).ljust(16).encode()
    s.sendall(size)
    s.sendall(data)


while True:

    try:
        cmd = s.recv(1024).decode().strip()

        # ---------------- FRAME REQUEST ----------------
        if cmd == "CAPTURE":
            frame = picam2.capture_array()
            send_frame(frame)

        # ---------------- RESULT ----------------
        elif cmd.startswith("ID"):

            parts = cmd.split("|")
            latest_id = parts[0].split(":")[1]
            latest_result = parts[1].split(":")[1]

            if latest_result == "good":
                good += 1
            elif latest_result == "no_cap":
                no_cap += 1
            elif latest_result == "no_label":
                no_label += 1

        # ---------------- DISPLAY ----------------
        frame = picam2.capture_array()
        h, w, _ = frame.shape

        cv2.putText(frame, "MYK AUTOMATION", (w//2 - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

        cv2.putText(frame, f"Good: {good}", (20, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"No Cap: {no_cap}", (20, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"No Label: {no_label}", (20, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        cv2.putText(frame, f"ID {latest_id} | {latest_result}",
                    (w-300, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Bottle System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception as e:
        print("[PI ERROR]", e)
        break

cv2.destroyAllWindows()
s.close()