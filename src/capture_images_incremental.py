from picamera2 import Picamera2
import cv2
import os
import time

# Class mapping
class_map = {
    "1": "Good",
    "2": "No_Cap",
    "3": "No_Label"
}

# Ask for metadata
brand = input("Enter brand (Nestle / Dasani / Aquafina): ").strip()
angle = input("Enter angle (horizontal / inclined): ").strip()

# Base path
base_path = os.path.expanduser("~/Bottle_Inspection/Images")

# Default class
current_class = "Good"

# Start camera ONCE
picam2 = Picamera2()
picam2.start()
time.sleep(2)

print("\nControls:")
print("1 = Good | 2 = No_Cap | 3 = No_Label")
print("s = save image | q = quit\n")

while True:
    frame = picam2.capture_array()
    
    # Show current class on screen
    cv2.putText(frame, f"Class: {current_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Camera Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    # Change class
    if chr(key) in class_map:
        current_class = class_map[chr(key)]
        print(f"Switched to: {current_class}")

    # Save image
    elif key == ord('s'):
        temp_folder = os.path.join(base_path, current_class, "Training_Sample")
        os.makedirs(temp_folder, exist_ok=True)

        existing_files = os.listdir(temp_folder)
        index = len(existing_files)

        filename = os.path.join(
            temp_folder,
            f"{brand}_{current_class}_{angle}_img_{index}.jpg"
        )

        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    # Quit
    elif key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()