import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
DATASET_PATH = "dataset"
IMG_SIZE = 128

data = []
labels = []

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    label = folder
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image: {img_path} - {e}")

X = np.array(data)
y = np.array(labels)

print("Dataset shape:", X.shape)
print("Unique labels:", np.unique(y))

# Encode labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize images
X = X / 255.0

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Label example:", y_train[0])
