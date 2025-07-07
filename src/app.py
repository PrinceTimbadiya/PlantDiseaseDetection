import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -------- CONFIG --------
MODEL_PATH = (
    "D:/AI-ML-Projects/PlantDiseaseDetection/saved_model/plant_disease_model.h5"
)
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 50.0  # lowered for practical usage

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


# -------- Load Model --------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_model()

# -------- Streamlit App --------
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image and get disease prediction!")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    # Preprocess
    img = np.array(image)
    if img.shape[2] == 4:
        img = img[:, :, :3]  # remove alpha channel if present

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("Predict"):
        pred = model.predict(img)
        pred_class_idx = np.argmax(pred)
        pred_class_name = CLASSES[pred_class_idx]
        confidence = np.max(pred) * 100

        st.success(f"**Prediction:** {pred_class_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ Note: Confidence is low. The result may not be accurate. "
                f"Please ensure you upload a clear leaf image."
            )
