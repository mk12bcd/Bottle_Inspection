
from picamera2 import Picamera2
import cv2
import os
import time

category = "Good Bottles"
save_path = f"Bottle_Inspection/Images/{category}/Training Sample"
os.makedirs(save_path,exist_ok = True)


picam2 = Picamera2()
picam2.start()
time.sleep(2)
count = 0
print("Press 's' to save, 'q' to quit")
while True:
	frame = picam2.capture_array()
	cv2.imshow("Camera Preview", frame) 
	key = cv2.waitKey(1) & 0xFF
	if key == ord('s'):
		filename = os.path.join(save_path, f"img_{count}.jpg")
		cv2.imwrite(filename, frame)
		print(f"Saved: {filename}")
		count += 1
	elif key == ord('q'):
		break
cv2.destroyAllWindows
picam2.stop()