from picamera2 import Picamera2
import socket
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

s = socket.socket()
s.connect(("192.168.1.2", 5000))

print("[PI] Connected")

# ---------------- STATE ----------------
latest_text = "Waiting"
latest_id = "-"

good_count = 0
no_cap_count = 0
no_label_count = 0

last_received = ""  # prevents double counting

while True:

    frame = picam2.capture_array()

    # ---------------- RECEIVE RESULT ----------------
    s.setblocking(False)
    try:
        data = s.recv(1024).decode()

        if "ID" in data and data != last_received:
            last_received = data

            parts = data.split("|")
            latest_id = parts[0].split(":")[1]
            latest_text = parts[1].split(":")[1]

            # update counters
            if latest_text == "good":
                good_count += 1
            elif latest_text == "no_cap":
                no_cap_count += 1
            elif latest_text == "no_label":
                no_label_count += 1

    except:
        pass

    s.setblocking(True)

    display = frame.copy()

    h, w, _ = display.shape

    # ---------------- TITLE ----------------
    cv2.putText(display, "MYK AUTOMATION",
                (int(w/2) - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 255), 3)

    # ---------------- COUNTS (BOTTOM LEFT) ----------------
    cv2.putText(display, f"Good: {good_count}", (20, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(display, f"No Cap: {no_cap_count}", (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(display, f"No Label: {no_label_count}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # ---------------- RESULT COLOR ----------------
    if latest_text == "good":
        color = (0, 255, 0)
        result_text = "GOOD"
    elif latest_text == "no_cap":
        color = (0, 0, 255)
        result_text = "NO CAP"
    elif latest_text == "no_label":
        color = (0, 165, 255)
        result_text = "NO LABEL"
    else:
        color = (200, 200, 200)
        result_text = "WAITING"

    # ---------------- BOTTOM RIGHT (BIG RESULT) ----------------
    text = f"ID {latest_id} | {result_text}"

    (tw, th), _ = cv2.getTextSize(text,
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 1, 3)

    cv2.putText(display,
                text,
                (w - tw - 20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    # ---------------- SHOW ----------------
    cv2.imshow("Bottle Inspection", display)

    # ---------------- CAPTURE HANDLER ----------------
    cmd = s.recv(16).decode().strip()

    if cmd == "CAPTURE":
        for _ in range(5):

            frame = picam2.capture_array()

            _, buffer = cv2.imencode(".jpg", frame)
            data = buffer.tobytes()

            size = str(len(data)).ljust(16).encode()

            s.sendall(size)
            s.sendall(data)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
s.close()