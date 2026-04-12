from picamera2 import Picamera2
import cv2
import os
import time

# Ask user what type of images to capture
mode = input("Enter mode (background / overlap): ").strip().lower()

if mode not in ["background", "overlap"]:
    print("Invalid mode. Use 'background' or 'overlap'")
    exit()

# Define save path
base_path = os.path.expanduser("~/Bottle_Inspection/Extra_Images")
save_folder = os.path.join(base_path, mode)

# Create folder if not exists
os.makedirs(save_folder, exist_ok=True)

# Start camera
picam2 = Picamera2()
picam2.start()
time.sleep(2)

print("\nPress 's' to save image")
print("Press 'q' to quit\n")

# Start count from existing images
count = len(os.listdir(save_folder))

while True:
    frame = picam2.capture_array()

    # Display mode on screen
    cv2.putText(frame, f"Mode: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture Extra Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = os.path.join(save_folder, f"{mode}_{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()