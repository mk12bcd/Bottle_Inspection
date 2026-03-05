 # main.py
import cv2
import os
import numpy as np

# ========================
# Step 0: Setup
# ========================
base_folder = os.path.join("..", "Images")
categories = ["Good Bottles", "Defective Bottles - Missing Cap", "Defective Bottles - No Label","Defective Bottles - Torn Label"]

X = []  # preprocessed images
y = []  # labels

# ========================
# Step 1: Loop through all categories for display and preprocessing
# ========================
for idx, category in enumerate(categories):
    folder_path = os.path.join(base_folder, category, "Testing Sample")
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        continue

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif'))]
    print(f"\nCategory: {category}, {len(files)} images found.")

    for f in files:
        img_path = os.path.join(folder_path, f)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {f}")
            continue

        # ========================
        # Display the image briefly for verification
        # ========================
        gray_display = cv2.cvtColor(cv2.resize(img, (400, 400)), cv2.COLOR_BGR2GRAY)
        cv2.imshow("Preview", gray_display)
        cv2.waitKey(200)  # show for 200 ms

        # ========================
        # Preprocess for dataset
        # ========================
        img_resized = cv2.resize(img, (200, 200))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        X.append(np.array(gray))
        y.append(idx)

cv2.destroyAllWindows()

# ========================
# Step 2: Convert to NumPy arrays and save
# ========================
X = np.array(X)
y = np.array(y)

print(f"\nTotal images processed: {len(X)}")
print(f"Total labels processed: {len(y)}")

# Save preprocessed data for later use
np.save("X.npy", X)
np.save("y.npy", y)
print("Saved preprocessed data to X.npy and y.npy")