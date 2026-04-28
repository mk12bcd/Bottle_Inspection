from picamera2 import Picamera2
import cv2
import os
import time

# ✅ ABSOLUTE PATH to your repo on Raspberry Pi
base_path = "/home/pi/Bottle_Inspection/Final_Images/Raw_Images"

# Create category folders if they don't exist
categories = {
    'g': "Good",
    'c': "No_cap",
    'l': "No_label"
}

for cat in categories.values():
    os.makedirs(os.path.join(base_path, cat), exist_ok=True)

picam2 = Picamera2()
picam2.start()
time.sleep(2)

count = 0

print("Press:")
print(" g → Save to Good")
print(" c → Save to No_cap")
print(" l → Save to No_label")
print(" q → Quit")

while True:
    frame = picam2.capture_array()
    cv2.imshow("Camera Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord('g'), ord('c'), ord('l')]:
        category = categories[chr(key)]
        save_path = os.path.join(base_path, category)

        filename = os.path.join(save_path, f"{category}_{count}.jpg")
        cv2.imwrite(filename, frame)

        print(f"Saved to {category}: {filename}")
        count += 1

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()