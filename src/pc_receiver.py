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

# Classification model - fallback only; identifies Good/No_cap/No_label on bottle crops
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

# ── FIX #2: Raised from 1 → 5 so count only updates after 5 identical frames.
# COUNT_STABLE_FRAMES = 1 caused constant 2↔3 oscillation every single frame.
COUNT_STABLE_FRAMES = 5

MAX_EXPECTED_BOTTLES = 3

SINGLE_BOTTLE_PX = 80
OVERLAP_RATIO_THRESHOLD = 1.7

TRIGGER_X_FRAC = 0.75
# ── FIX #3: Kept at 5 — coder explicitly requires 5 frames for majority vote.
MIN_HISTORY_FRAMES = 5
# ── FIX #4: Raised from 0.50 → 0.65 — require a clearer majority vote.
MIN_VOTE_FRAC = 0.65
SPATIAL_RADIUS_PX = 80
SPATIAL_COOLDOWN_S = 3.0
TRACK_CLEANUP_AGE = 60

# ─────────────────────────────────────────────────────────────
# Vision rule settings for this exact conveyor/bottle setup
# ─────────────────────────────────────────────────────────────
# Camera view is end-on/down-belt: the front bottle is usually the one
# closest to the camera, so it has the largest bottom y-coordinate.
FRONT_SELECTION_MODE = "bottom_y"

# Use simple visual checks first because the defects are visually simple:
# blue cap at the top + coloured label around the middle.
VISUAL_RULE_ENABLED = True
MODEL_FALLBACK_ENABLED = True

# Blue cap detection threshold. This uses the largest blue blob in the
# top/neck region. A real cap is a filled blue disk; a missing cap may still
# show a small blue ring, so we require a larger blue contour.
CAP_BLUE_LOW = np.array([85, 50, 40])
CAP_BLUE_HIGH = np.array([135, 255, 255])
CAP_CONTOUR_MIN_RATIO = 0.10
CAP_UNCERTAIN_MARGIN = 0.02

# Label detection threshold. The label has strong blue/red/pink colour in
# the middle region. Clear/no-label bottles should stay below this ratio.
LABEL_BLUE_LOW = np.array([85, 60, 40])
LABEL_BLUE_HIGH = np.array([135, 255, 255])
LABEL_RED1_LOW = np.array([0, 60, 40])
LABEL_RED1_HIGH = np.array([20, 255, 255])
LABEL_RED2_LOW = np.array([145, 60, 40])
LABEL_RED2_HIGH = np.array([179, 255, 255])
LABEL_COLOR_MIN_RATIO = 0.08
LABEL_UNCERTAIN_MARGIN = 0.025

# Keep the model threshold at 70% as requested by the coder.
MODEL_CONF_THRESHOLD = 0.70

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


def crop_with_padding(frame, x1, y1, x2, y2, pad_x_frac=0.15, pad_y_frac=0.18):
    """Return a padded bottle crop and its clipped coordinates."""
    h_frame, w_frame = frame.shape[:2]

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_x = int(box_w * pad_x_frac)
    pad_y = int(box_h * pad_y_frac)

    x1c = max(0, int(x1) - pad_x)
    y1c = max(0, int(y1) - pad_y)
    x2c = min(w_frame, int(x2) + pad_x)
    y2c = min(h_frame, int(y2) + pad_y)

    crop = frame[y1c:y2c, x1c:x2c]
    return crop, (x1c, y1c, x2c, y2c)


def largest_contour_ratio(mask):
    """Largest contour area divided by mask area."""
    area = mask.shape[0] * mask.shape[1]
    if area <= 0:
        return 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    return max(cv2.contourArea(c) for c in contours) / float(area)


def model_classify_crop(crop):
    """YOLO model fallback: classify the crop only, never the full frame."""
    try:
        results = classification_model(
            crop,
            verbose=False,
            conf=MODEL_CONF_THRESHOLD
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None, 0.0

        best_idx = results.boxes.conf.argmax().item()
        class_id = int(results.boxes.cls[best_idx])
        conf = float(results.boxes.conf[best_idx].item())
        label = classification_model.names[class_id]

        return label, conf

    except Exception as e:
        print(f"[MODEL CLASSIFY ERROR] {e}")
        return None, 0.0


def visual_cap_label_classify(crop):
    """
    Rule-based classification for the bottle crop:
      - cap region = top part of the crop
      - label region = middle part of the crop

    Returns:
      final_label, cap_present, label_present, cap_score, label_score, uncertain
    """
    h, w = crop.shape[:2]

    if h < 40 or w < 30:
        return None, False, False, 0.0, 0.0, True

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # ── Cap check ────────────────────────────────────────────────
    # Use only central top area so side walls/background do not affect the cap.
    cap_y1 = 0
    cap_y2 = max(1, int(h * 0.30))
    cap_x1 = int(w * 0.20)
    cap_x2 = int(w * 0.80)
    cap_region = hsv[cap_y1:cap_y2, cap_x1:cap_x2]

    cap_mask = cv2.inRange(cap_region, CAP_BLUE_LOW, CAP_BLUE_HIGH)
    cap_score = largest_contour_ratio(cap_mask)
    cap_present = cap_score >= CAP_CONTOUR_MIN_RATIO

    # ── Label check ──────────────────────────────────────────────
    # Middle region contains the wrapper. We look for strong blue/red/pink.
    label_y1 = int(h * 0.28)
    label_y2 = int(h * 0.68)
    label_x1 = int(w * 0.05)
    label_x2 = int(w * 0.95)
    label_region = hsv[label_y1:label_y2, label_x1:label_x2]

    label_blue = cv2.inRange(label_region, LABEL_BLUE_LOW, LABEL_BLUE_HIGH)
    label_red1 = cv2.inRange(label_region, LABEL_RED1_LOW, LABEL_RED1_HIGH)
    label_red2 = cv2.inRange(label_region, LABEL_RED2_LOW, LABEL_RED2_HIGH)
    label_mask = cv2.bitwise_or(label_blue, cv2.bitwise_or(label_red1, label_red2))

    label_score = cv2.countNonZero(label_mask) / float(label_mask.shape[0] * label_mask.shape[1])
    label_present = label_score >= LABEL_COLOR_MIN_RATIO

    # If the measured score is close to a threshold, let the model help.
    cap_uncertain = abs(cap_score - CAP_CONTOUR_MIN_RATIO) <= CAP_UNCERTAIN_MARGIN
    label_uncertain = abs(label_score - LABEL_COLOR_MIN_RATIO) <= LABEL_UNCERTAIN_MARGIN
    uncertain = cap_uncertain or label_uncertain

    # Pi code currently accepts only Good, No_cap, and No_label.
    # If both cap and label are missing, classify it as No_cap so it is rejected.
    if cap_present and label_present:
        final_label = "Good"
    elif not cap_present:
        final_label = "No_cap"
    else:
        final_label = "No_label"

    return final_label, cap_present, label_present, cap_score, label_score, uncertain


def classify_bottle_crop(frame, x1, y1, x2, y2):
    """
    Final bottle classification.

    Main fixes added:
      1. crop one bottle only, not the full frame
      2. detect cap and label visually from the crop
      3. use YOLO model only as a fallback when the visual rule is uncertain
      4. keep model confidence threshold at 70%
    """
    try:
        crop, _ = crop_with_padding(frame, x1, y1, x2, y2)

        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        visual_label = None
        cap_present = False
        label_present = False
        cap_score = 0.0
        label_score = 0.0
        uncertain = True

        if VISUAL_RULE_ENABLED:
            (
                visual_label,
                cap_present,
                label_present,
                cap_score,
                label_score,
                uncertain,
            ) = visual_cap_label_classify(crop)

        model_label = None
        model_conf = 0.0

        # Use the model when visual scores are borderline, or if the visual rule is off.
        if MODEL_FALLBACK_ENABLED and (uncertain or not VISUAL_RULE_ENABLED):
            model_label, model_conf = model_classify_crop(crop)

        if visual_label is None and model_label is None:
            return None

        # Decision rule:
        # - If visual check is confident, trust it because the defects are simple.
        # - If visual check is uncertain and model is confident, use the model.
        # - Otherwise use the visual result so the bottle still gets classified/rejected.
        if uncertain and model_label is not None:
            final_label = model_label
            source = f"MODEL {model_conf:.2f}"
        else:
            final_label = visual_label
            source = "VISION"

        print(
            f"[CLASSIFY] {final_label} via {source} | "
            f"cap={'Y' if cap_present else 'N'}({cap_score:.3f}) "
            f"label={'Y' if label_present else 'N'}({label_score:.3f})"
        )

        return final_label

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
        # STEP 4a: Find the FRONT bottle
        # Current camera is end-on/down-belt, so the bottle closest to the
        # camera normally has the largest bottom y-coordinate (y2).
        # This replaces the old smallest-cx rule, which was correct only for
        # a side-view right→left conveyor.
        # ─────────────────────────────────────────────────────────────
        closest_track_id = None
        best_front_score = -float('inf')

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            # Skip already classified bottles
            if track_id in sent_ids:
                continue

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            box_area = max(1, (x2 - x1) * (y2 - y1))

            if FRONT_SELECTION_MODE == "bottom_y":
                # Larger y2 = lower in image = closer to the camera/front.
                front_score = y2 + 0.0005 * box_area
            elif FRONT_SELECTION_MODE == "area":
                # Backup option: closest bottle is often the largest one.
                front_score = box_area
            else:
                # Old side-view fallback: smallest x-centre.
                cx = int((x1 + x2) / 2)
                front_score = -cx

            if front_score > best_front_score:
                best_front_score = front_score
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

            # Keep trigger-line crossing detection for side-on camera setups.
            # NOTE: for the current end-on camera (belt moves toward camera,
            # cy changes rather than cx), the trigger gate below has been
            # REMOVED so results still reach the Pi.
            prev_cx = track_last_cx.get(track_id)
            if prev_cx is not None and prev_cx > trigger_px >= cx:
                track_triggered.add(track_id)
            track_last_cx[track_id] = cx

            if track_id not in track_history:
                track_history[track_id] = []

            # Classify only the FRONT bottle each frame
            if track_id == closest_track_id:
                label = classify_bottle_crop(frame, x1, y1, x2, y2)

                if label is not None:
                    track_history[track_id].append(label)

            # Keep up to 10 history frames
            if len(track_history[track_id]) > 10:
                track_history[track_id].pop(0)

            history = track_history[track_id]

            # Skip if already sent
            if track_id in sent_ids:
                continue

            # ── FIX #5 (TRIGGER REMOVAL):
            # Previous code required the bottle to cross an X-axis trigger line
            # (prev_cx > trigger_px >= cx).  The camera faces DOWN the length
            # of the belt (end-on), so bottles move toward the camera — their
            # Y changes, not their X.  cx stays near the frame centre and never
            # crosses the trigger, so ID results were NEVER sent to the Pi.
            #
            # New rule: send as soon as the FRONT bottle has accumulated
            # MIN_HISTORY_FRAMES classified frames with a conclusive majority.
            # No trigger line required.

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

        # Trigger line is only displayed for reference; classification sending no longer
        # depends on this X-line because the current camera view is end-on/down-belt.
        cv2.line(frame, (trigger_px, 0), (trigger_px, h), (0, 255, 255), 1)
        cv2.putText(
            frame,
            "REF LINE",
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
