import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# -------- CONFIG --------
MODEL_PATH = "saved_model/plant_disease_model.h5"  # SAME .h5 MODEL
IMG_SIZE = 128  # SAME AS TRAINING
CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# -------- 1) Load Model --------
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# -------- 2) Get Image Path (Dynamic) --------

# ✅ OPTION A: Command Line Argument
if len(sys.argv) > 1:
    TEST_IMAGE_PATH = sys.argv[1]
    print(f"[INFO] Using image path from command line: {TEST_IMAGE_PATH}")

# ✅ OPTION B: input() Prompt
else:
    TEST_IMAGE_PATH = input("Enter leaf image path (e.g., test_leaf.jpg): ")
    print(f"[INFO] Using image path from input: {TEST_IMAGE_PATH}")

# -------- 3) Load Test Image --------
try:
    img = cv2.imread(TEST_IMAGE_PATH)
    if img is None:
        raise Exception("Image not found or cannot be opened!")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Batch dimension

    # -------- 4) Predict --------
    pred = model.predict(img)
    pred_class_idx = np.argmax(pred)
    pred_class_name = CLASSES[pred_class_idx]

    print(f"[INFO] Predicted Class: {pred_class_name}")
    print(f"[INFO] Confidence: {np.max(pred) * 100:.2f}%")

except Exception as e:
    print(f"[ERROR] {e}")
