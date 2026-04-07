from picamera2 import Picamera2
import cv2
import os
import time

# Classes
classes = ["Good", "No_Cap", "No_Label"]

# Ask user for brand and angle ONCE
brand = input("Enter brand (Nestle / Dasani / Aquafina): ").strip()
angle = input("Enter angle (horizontal / inclined): ").strip()

for category in classes:
    # Temp folder for today
    base_path = os.path.expanduser("~/Bottle_Inspection/Images")
    temp_folder = os.path.join(base_path, category, "Today")
    os.makedirs(temp_folder, exist_ok=True)

    # Main folder (old + new images)
    main_folder = os.path.join(base_path, category, "Training_Sample")
    os.makedirs(main_folder, exist_ok=True)

    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)

    count = 0
    print(f"\n--- Capturing for class: {category} ---")
    print("Press 's' to save, 'q' to move to next class")

    while True:
        frame = picam2.capture_array()
        cv2.imshow(f"{category} Preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            filename = os.path.join(
                temp_folder,
                f"{brand}_{category}_{angle}_img_{count}.jpg"
            )
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            count += 1

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

    # Merge into main folder with continuous numbering
    existing_files = os.listdir(main_folder)
    index = len(existing_files)

    for f in sorted(os.listdir(temp_folder)):
        src = os.path.join(temp_folder, f)
        new_name = f"{brand}_{category}_{angle}_img_{index}.jpg"
        dst = os.path.join(main_folder, new_name)
        os.rename(src, dst)
        index += 1

    os.rmdir(temp_folder)

    print(f"Finished {category}. Total images now: {index}")