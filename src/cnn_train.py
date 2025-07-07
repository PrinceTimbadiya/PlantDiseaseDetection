import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# -------------------------------
# 1) Load Data (Same as main.py)
# -------------------------------
DATASET_PATH = "dataset"
IMG_SIZE = 128

data = []
labels = []

print("[INFO] Loading images...")

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
        except:
            pass  # skip corrupt images

X = np.array(data)
y = np.array(labels)

print("[INFO] Images loaded:", X.shape)
print("[INFO] Labels:", np.unique(y))

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = X / 255.0  # normalize

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -------------------------------
# 2) CNN Architecture
# -------------------------------
print("[INFO] Building CNN model...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# -------------------------------
# 3) Train Model
# -------------------------------
print("[INFO] Training CNN model...")

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    X_train, y_train,
    epochs=10,  # Start with 5-10 for testing
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -------------------------------
# 4) Plot Accuracy/Loss
# -------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# 5) Save Model
# -------------------------------
SAVE_PATH = "saved_model/plant_disease_model.h5"
model.save(SAVE_PATH)
print(f"[INFO] Model saved to: {SAVE_PATH}")
