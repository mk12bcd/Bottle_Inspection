import socket
import numpy as np
import cv2
import math
import struct
import threading
import queue
import time
from collections import Counter
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Detection model - fast, general bottle detection (COCO pretrained)
detection_model = YOLO("yolov8s.pt")

# Classification model - identifies Good/No_cap/No_label on full frames
classification_model = YOLO(r"C:\Users\mubar\OneDrive - Universities of Canada\Documents\Bottle_Inspection\models\best_fold5.pt")

tracker = DeepSort(
    max_age=3,
    n_init=2,
    max_cosine_distance=0.3,
    nn_budget=100
)

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", 5000))
server.listen(1)

print("Waiting for Pi to connect...")
conn, addr = server.accept()
conn.settimeout(5)
print(f"Pi connected from {addr}")

HEADER = 4

COUNT_STABLE_FRAMES = 1
MAX_EXPECTED_BOTTLES = 3

SINGLE_BOTTLE_PX = 80
OVERLAP_RATIO_THRESHOLD = 1.7

TRIGGER_X_FRAC = 0.75          # CHANGED: moved right for right→left belt
MIN_HISTORY_FRAMES = 5         # CHANGED: 5 frames for stable classification
MIN_VOTE_FRAC = 0.50           # CHANGED: lowered from 0.60
SPATIAL_RADIUS_PX = 80
SPATIAL_COOLDOWN_S = 3.0
TRACK_CLEANUP_AGE = 60

frame_queue = queue.Queue(maxsize=1)
running_capture = True

track_history = {}
sent_ids = set()
track_last_cx = {}
track_triggered = set()
recently_counted = []
track_last_seen = {}
frame_count = 0

fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0.0


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def recv_msg(sock):
    raw_len = recv_exact(sock, HEADER)
    if raw_len is None:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    payload = recv_exact(sock, msg_len)
    return payload.decode() if payload else None


def send_msg(sock, text):
    encoded = text.encode()
    sock.sendall(struct.pack(">I", len(encoded)) + encoded)


def recv_frame(sock):
    raw_len = recv_exact(sock, HEADER)
    if raw_len is None:
        return None
    frame_len = struct.unpack(">I", raw_len)[0]
    data = recv_exact(sock, frame_len)
    if data is None:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


count_frame_buffer = []
last_sent_count = -1

good_count = 0
no_cap_count = 0
no_label_count = 0
latest_class = "waiting"


def capture_thread():
    global running_capture

    while running_capture:
        try:
            send_msg(conn, "CAPTURE")
            frame = recv_frame(conn)

            if frame is None:
                running_capture = False
                break

            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait(frame)

        except Exception as e:
            print(f"Capture thread error: {e}")
            running_capture = False
            break


threading.Thread(target=capture_thread, daemon=True).start()


def classify_bottle_crop(frame, x1, y1, x2, y2):
    """
    Pass full frame resized to 832x832 (training size) to classification model.
    Returns: Good, No_cap, No_label or None.
    """
    try:
        # CHANGED: use full frame resized to training size, not crop
        full_frame_resized = cv2.resize(frame, (832, 832))

        results = classification_model(full_frame_resized, verbose=False, conf=0.1)[0]

        # CHANGED: use boxes (detection model), not probs
        if results.boxes is None or len(results.boxes) == 0:
            return None

        best_idx = results.boxes.conf.argmax().item()
        class_id = int(results.boxes.cls[best_idx])
        conf = results.boxes.conf[best_idx].item()
        label = classification_model.names[class_id]

        print(f"[CLASSIFY] {label} ({conf:.2f})")

        if conf < 0.25:
            return None

        return label

    except Exception as e:
        print(f"[CLASSIFY ERROR] {e}")
        return None


def analyse_detections(boxes, frame):
    if boxes is None or len(boxes) == 0:
        return []

    detections = []

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())
        if class_id != 39:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        conf = boxes.conf[i].item()

        detections.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'conf': conf,
            'width': x2 - x1
        })

    return detections


try:
    while True:
        frame_count += 1

        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            if not running_capture:
                break
            continue

        # ─────────────────────────────────────────────────────────────
        # STEP 1: Fast bottle detection using YOLOv8s
        # ─────────────────────────────────────────────────────────────
        detection_results = detection_model(frame, conf=0.50, classes=[39])[0]
        detection_boxes = detection_results.boxes
        detections = analyse_detections(detection_boxes, frame)

        raw_count = len(detections)

        count_frame_buffer.append(raw_count)

        if len(count_frame_buffer) > COUNT_STABLE_FRAMES:
            count_frame_buffer.pop(0)

        if (
            len(count_frame_buffer) == COUNT_STABLE_FRAMES
            and len(set(count_frame_buffer)) == 1
        ):
            stable_count = count_frame_buffer[0]

            if stable_count != last_sent_count:

                if stable_count > MAX_EXPECTED_BOTTLES:
                    print(
                        f"[COUNT WARN] Stable count={stable_count} exceeds "
                        f"MAX_EXPECTED_BOTTLES={MAX_EXPECTED_BOTTLES}"
                    )
                    stable_count = MAX_EXPECTED_BOTTLES

                send_msg(conn, f"COUNT:{stable_count}")
                print(f"[COUNT STABLE] sending COUNT:{stable_count}")

                last_sent_count = stable_count

        # ─────────────────────────────────────────────────────────────
        # STEP 2: Prepare detections for DeepSort tracking
        # ─────────────────────────────────────────────────────────────
        detections_for_tracker = []
        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            w = x2 - x1
            h = y2 - y1
            conf = det['conf']

            detections_for_tracker.append(
                ([x1, y1, w, h], conf, "bottle")
            )

        # ─────────────────────────────────────────────────────────────
        # STEP 3: Update DeepSort tracker
        # ─────────────────────────────────────────────────────────────
        tracks = tracker.update_tracks(
            detections_for_tracker,
            frame=frame
        )

        h, w, _ = frame.shape
        trigger_px = int(w * TRIGGER_X_FRAC)

        # ─────────────────────────────────────────────────────────────
        # STEP 4a: Find the FRONT bottle (smallest cx = furthest left)
        # Belt moves right→left so front bottle has smallest cx
        # ─────────────────────────────────────────────────────────────
        closest_track_id = None
        closest_distance_to_trigger = float('inf')

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            # Skip already classified bottles
            if track_id in sent_ids:
                continue

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx = int((x1 + x2) / 2)

            # CHANGED: pick smallest cx = front bottle on right→left belt
            if cx < closest_distance_to_trigger:
                closest_distance_to_trigger = cx
                closest_track_id = track_id

        # ─────────────────────────────────────────────────────────────
        # STEP 4b: Process confirmed tracks
        # ─────────────────────────────────────────────────────────────
        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id
            track_last_seen[track_id] = frame_count

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # CHANGED: right→left crossing detection (cx decreasing)
            prev_cx = track_last_cx.get(track_id)
            if prev_cx is not None and prev_cx > trigger_px >= cx:
                track_triggered.add(track_id)
            track_last_cx[track_id] = cx

            if track_id not in track_history:
                track_history[track_id] = []

            # Classify only the FRONT bottle
            if track_id == closest_track_id:
                label = classify_bottle_crop(frame, x1, y1, x2, y2)

                if label is not None:
                    track_history[track_id].append(label)

            # CHANGED: keep up to 10 history frames
            if len(track_history[track_id]) > 10:
                track_history[track_id].pop(0)

            history = track_history[track_id]

            # Skip if already sent
            if track_id in sent_ids:
                continue

            # Skip if not crossed trigger line
            if track_id not in track_triggered:
                continue

            # Need enough history for stable classification
            if len(history) < MIN_HISTORY_FRAMES:
                continue

            # Vote on most common classification
            final, win_count = Counter(history).most_common(1)[0]

            if win_count / len(history) < MIN_VOTE_FRAC:
                continue

            # Check for spatial duplicates
            now = time.time()
            recently_counted = [
                (old_cx, old_cy, ts)
                for old_cx, old_cy, ts in recently_counted
                if now - ts <= SPATIAL_COOLDOWN_S * 2
            ]

            is_duplicate = any(
                math.hypot(cx - old_cx, cy - old_cy) <= SPATIAL_RADIUS_PX
                and now - ts <= SPATIAL_COOLDOWN_S
                for old_cx, old_cy, ts in recently_counted
            )

            if is_duplicate:
                sent_ids.add(track_id)
                print(f"[DEDUP] Skipping duplicate track ID:{track_id}")
                continue

            recently_counted.append((cx, cy, now))

            latest_class = final

            if final == "Good":
                good_count += 1
            elif final == "No_cap":
                no_cap_count += 1
            elif final == "No_label":
                no_label_count += 1

            send_msg(conn, f"ID:{track_id}|{final}")
            print(f"[CLASSIFY] Sent classification → ID:{track_id}|{final}")

            sent_ids.add(track_id)

        # ─────────────────────────────────────────────────────────────
        # STEP 5: Cleanup stale tracks
        # ─────────────────────────────────────────────────────────────
        stale_track_ids = [
            track_id
            for track_id, last_seen in track_last_seen.items()
            if frame_count - last_seen > TRACK_CLEANUP_AGE
        ]

        for track_id in stale_track_ids:
            track_history.pop(track_id, None)
            track_last_cx.pop(track_id, None)
            track_last_seen.pop(track_id, None)

        if detection_boxes is not None:
            for i in range(len(detection_boxes)):
                if int(detection_boxes.cls[i].item()) != 39:
                    continue
                x1, y1, x2, y2 = [int(v) for v in detection_boxes.xyxy[i].tolist()]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            x1, y1, x2, y2 = map(int, track.to_ltrb())

            box_color = (0, 255, 255) if track_id == closest_track_id else (0, 255, 0)
            box_thickness = 3 if track_id == closest_track_id else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

            history = track_history.get(track_id, [])
            if history:
                final, _ = Counter(history).most_common(1)[0]
                cv2.putText(
                    frame,
                    f"ID:{track_id} {final}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )
            else:
                status_text = f"ID:{track_id} ◄ CLASSIFYING" if track_id == closest_track_id else f"ID:{track_id}"
                cv2.putText(
                    frame,
                    status_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )

        cv2.line(frame, (trigger_px, 0), (trigger_px, h), (0, 255, 255), 1)
        cv2.putText(
            frame,
            "TRIGGER",
            (trigger_px + 4, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

        frames_agreed = (
            count_frame_buffer.count(raw_count)
            if count_frame_buffer
            else 0
        )

        count_color = (
            (0, 200, 0)
            if frames_agreed == COUNT_STABLE_FRAMES
            else (0, 165, 255)
        )

        cv2.putText(
            frame,
            f"Visible: {raw_count} Stable: {frames_agreed}/{COUNT_STABLE_FRAMES}",
            (w - 320, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            count_color,
            2,
        )

        text = "MYK AUTOMATION"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 45
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

        cv2.rectangle(frame, (10, h - 120), (320, h - 10), (255, 255, 255), -1)

        cv2.putText(
            frame,
            f"Good: {good_count}",
            (20, h - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 0),
            2,
        )

        cv2.putText(
            frame,
            f"No Cap: {no_cap_count}",
            (20, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"No Label: {no_label_count}",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 254),
            2,
        )

        status = f"Current: {latest_class}"

        cv2.rectangle(
            frame,
            (w - 300, h - 60),
            (w - 10, h - 10),
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            frame,
            status,
            (w - 290, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        cv2.imshow("Bottle Inspection - PC View", frame)

        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    print("Shutting down")

finally:
    running_capture = False
    conn.close()
    server.close()
    cv2.destroyAllWindows()