# train_classifier.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblibgit add .
git commit -m "Add scripts and images"
git push origin mai
# ========================
# Step 1: Load preprocessed data
# ========================
X = np.load("X.npy")
y = np.load("y.npy")

print("Data loaded:")
print("X shape:", X.shape)  # (num_images, 200, 200)
print("y shape:", y.shape)  # (num_images,)

# ========================
# Step 2: Flatten images for simple classifier
# ========================
X = X.reshape(-1, 200*200)  # flatten 2D images to 1D vectors

# ========================
# Step 3: Split into train and test sets
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ========================
# Step 4: Train Random Forest classifier
# ========================
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ========================
# Step 5: Evaluate on test set
# ========================
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ========================
# Step 6: Save trained model
# ========================
joblib.dump(clf, "bottle_classifier.pkl")
print("Classifier saved as bottle_classifier.pkl")