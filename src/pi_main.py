from picamera2 import Picamera2
import socket
import cv2
import RPi.GPIO as GPIO
import signal
import struct
import threading

# ─────────────────────────────────────────────────────────────────
# PIN ASSIGNMENTS
# ─────────────────────────────────────────────────────────────────
RELAY_CLASSIFY  = 17   # GPIO17 → reject/accept signal to PLC
RELAY_COUNT_MSB = 27   # GPIO27 → count bit 1 (MSB)
RELAY_COUNT_LSB = 22   # GPIO22 → count bit 0 (LSB)

RELAY_ON  = GPIO.HIGH
RELAY_OFF = GPIO.LOW

REJECT_PULSE_SEC = 0.2   # how long reject relay stays ON

# ─────────────────────────────────────────────────────────────────
# GPIO SETUP
# ─────────────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_CLASSIFY,  GPIO.OUT, initial=RELAY_OFF)
GPIO.setup(RELAY_COUNT_MSB, GPIO.OUT, initial=RELAY_OFF)
GPIO.setup(RELAY_COUNT_LSB, GPIO.OUT, initial=RELAY_OFF)

# ─────────────────────────────────────────────────────────────────
# CAMERA SETUP
# ─────────────────────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (640, 480)})
)
picam2.start()

# ─────────────────────────────────────────────────────────────────
# SOCKET SETUP
# ─────────────────────────────────────────────────────────────────
PC_IP = "192.168.1.7"
PORT  = 5000

s = socket.socket()
s.connect((PC_IP, PORT))
s.settimeout(30.0)

# ─────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────
good_count     = 0
no_cap_count   = 0
no_label_count = 0

current_count      = 0
processed_ids      = set()
classify_off_timer = None
relay_lock         = threading.Lock()

running = True
HEADER  = 4

# ─────────────────────────────────────────────────────────────────
# MESSAGING HELPERS
# ─────────────────────────────────────────────────────────────────
def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        data += chunk
    return data


def recv_msg(sock):
    raw_len = recv_exact(sock, HEADER)
    msg_len = struct.unpack(">I", raw_len)[0]
    return recv_exact(sock, msg_len).decode()


def send_msg(sock, text):
    encoded = text.encode()
    sock.sendall(struct.pack(">I", len(encoded)) + encoded)


def send_frame(sock):
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    data = buffer.tobytes()
    sock.sendall(struct.pack(">I", len(data)) + data)

# ─────────────────────────────────────────────────────────────────
# RELAY HELPERS
# ─────────────────────────────────────────────────────────────────
def set_count_relays(count):
    """
    Encode count (0-3) as 2-bit binary on two relay outputs.

    count | MSB(GPIO27) | LSB(GPIO22) | PLC reads
    ------+-------------+-------------+-----------
      0   |    OFF (0)  |   OFF (0)   |     0
      1   |    OFF (0)  |    ON (1)   |     1
      2   |     ON (1)  |   OFF (0)   |     2
      3   |     ON (1)  |    ON (1)   |     3
    """
    msb = RELAY_ON if (count & 0b10) else RELAY_OFF
    lsb = RELAY_ON if (count & 0b01) else RELAY_OFF

    GPIO.output(RELAY_COUNT_MSB, msb)
    GPIO.output(RELAY_COUNT_LSB, lsb)

    msb_str = "ON  (1)" if msb == RELAY_ON else "OFF (0)"
    lsb_str = "ON  (1)" if lsb == RELAY_ON else "OFF (0)"

    print(f"\n{'─'*60}")
    print(f"  [COUNT RELAY UPDATE]")
    print(f"    Bottles on belt : {count}")
    print(f"    GPIO27 MSB      : {msb_str}")
    print(f"    GPIO22 LSB      : {lsb_str}")
    print(f"    Binary sent     : {int(msb==RELAY_ON)}{int(lsb==RELAY_ON)}")
    print(f"    PLC reads       : {count} bottle(s)")
    print(f"{'─'*60}")


def _relay_off_callback():
    """Timer callback: de-energise classify relay after pulse."""
    global classify_off_timer
    with relay_lock:
        GPIO.output(RELAY_CLASSIFY, RELAY_OFF)
        classify_off_timer = None

    print(f"\n{'─'*60}")
    print(f"  [CLASSIFY RELAY]  GPIO17 → OFF  (reject pulse ended)")
    print(f"{'─'*60}")


def activate_reject_relay(result):
    """Energise classify relay and auto-off after REJECT_PULSE_SEC."""
    global classify_off_timer

    with relay_lock:
        if classify_off_timer is not None:
            classify_off_timer.cancel()

        GPIO.output(RELAY_CLASSIFY, RELAY_ON)

        print(f"\n{'─'*60}")
        print(f"  [CLASSIFY RELAY]  GPIO17 → ON   (REJECTING — {result})")
        print(f"  Auto-off in {REJECT_PULSE_SEC * 1000:.0f} ms")
        print(f"{'─'*60}")

        classify_off_timer = threading.Timer(REJECT_PULSE_SEC, _relay_off_callback)
        classify_off_timer.start()


def deactivate_classify_relay(result):
    """Good bottle — ensure relay is OFF (no rejection)."""
    global classify_off_timer

    with relay_lock:
        if classify_off_timer is not None:
            classify_off_timer.cancel()
            classify_off_timer = None

        GPIO.output(RELAY_CLASSIFY, RELAY_OFF)

    print(f"\n{'─'*60}")
    print(f"  [CLASSIFY RELAY]  GPIO17 → OFF  (PASS — {result})")
    print(f"{'─'*60}")

# ─────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────
def stop(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

# ─────────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  Raspberry Pi – Bottle Inspection Node")
print(f"  Connected to PC at {PC_IP}:{PORT}")
print(f"  Relay pins  →  CLASSIFY=GPIO{RELAY_CLASSIFY}  "
      f"MSB=GPIO{RELAY_COUNT_MSB}  LSB=GPIO{RELAY_COUNT_LSB}")
print(f"  Reject pulse duration: {REJECT_PULSE_SEC * 1000:.0f} ms")
print("  Waiting for commands from PC...")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────
while running:

    try:
        cmd = recv_msg(s)
    except Exception as e:
        print(f"\n[ERROR] recv_msg failed: {e}")
        break

    # ── FRAME REQUEST ─────────────────────────────────────────────
    if cmd == "CAPTURE":
        try:
            send_frame(s)
        except Exception as e:
            print(f"[ERROR] send_frame failed: {e}")
            break

    # ── BOTTLE COUNT UPDATE ───────────────────────────────────────
    elif cmd.startswith("COUNT:"):
        try:
            count = int(cmd.split(":")[1])
            count = min(count, 3)
        except Exception as e:
            print(f"[ERROR] COUNT parse failed: {e}")
            continue

        if count != current_count:
            print(f"\n{'═'*60}")
            print(f"  [PC → Pi] COUNT received: {current_count} → {count} bottle(s)")
            print(f"{'═'*60}")

            current_count = count
            set_count_relays(current_count)

    # ── CLASSIFICATION RESULT ─────────────────────────────────────
    elif cmd.startswith("ID:"):
        try:
            parts     = cmd.split("|")
            bottle_id = int(parts[0].split(":")[1])
            result    = parts[1]
        except Exception as e:
            print(f"[ERROR] ID parse failed: {e}")
            continue

        if bottle_id in processed_ids:
            print(f"  [SKIP] Bottle #{bottle_id} already processed.")
            continue

        processed_ids.add(bottle_id)

        # ── DISPLAY FINAL CLASS ───────────────────────────────────
        print(f"\n{'═'*60}")
        print(f"  [PC → Pi] Classification received")
        print(f"    Bottle ID    : #{bottle_id}")
        print(f"    Final class  : {result}")
        print(f"{'═'*60}")

        # ── DRIVE RELAY BASED ON CLASS ────────────────────────────
        if result == "Good":
            good_count += 1
            deactivate_classify_relay(result)

        elif result in ("No_cap", "No_label"):
            if result == "No_cap":
                no_cap_count += 1
            else:
                no_label_count += 1
            activate_reject_relay(result)

        else:
            print(f"  [WARN] Unknown class '{result}' — no relay action taken.")

        # ── RUNNING TOTALS ────────────────────────────────────────
        print(f"\n  [TOTALS]")
        print(f"    Good     : {good_count}")
        print(f"    No cap   : {no_cap_count}")
        print(f"    No label : {no_label_count}")
        print(f"    Total    : {good_count + no_cap_count + no_label_count}")
        print(f"{'─'*60}")

    else:
        print(f"[WARN] Unknown command received: '{cmd}'")

# ─────────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Shutting down — turning all relays OFF...")
print("=" * 60)

if classify_off_timer is not None:
    classify_off_timer.cancel()

with relay_lock:
    GPIO.output(RELAY_CLASSIFY,  RELAY_OFF)
    GPIO.output(RELAY_COUNT_MSB, RELAY_OFF)
    GPIO.output(RELAY_COUNT_LSB, RELAY_OFF)

GPIO.cleanup()
picam2.stop()
s.close()

print("[Pi] Shutdown complete.")