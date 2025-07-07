# 🌿 Plant Disease Detection

This is a simple AI/ML project that detects diseases in plant leaf images using a Convolutional Neural Network (CNN) built with TensorFlow. The trained model is deployed using a Streamlit web app, so users can easily upload leaf images and get disease predictions.

---

## 📁 Project Structure
PlantDiseaseDetection/
│
├── dataset/ # Leaf images for training/testing
├── saved_model/ # Trained .h5 model file
│ └── plant_disease_model.h5
├── src/ # Source code
│ ├── app.py # Streamlit GUI app
│ ├── cnn_train.py # CNN training script
│ ├── main.py # Data preprocessing / helper script
│ └── predict.py # Standalone predict script
├── test_leaf.jpg # Sample test image
└── venv/ # Virtual environment (not pushed)


## ✅ How to Run

**1️⃣ Clone the repo:**
```bash
git clone https://github.com/YOUR-USERNAME/PlantDiseaseDetection.git
cd PlantDiseaseDetection

2️⃣ Create virtual environment:
python -m venv venv

3️⃣ Activate virtual env:
.\venv\Scripts\activate

4️⃣ Install dependencies:
pip install -r requirements.txt

5️⃣ Run Streamlit app:
streamlit run src/app.py


⚠️ Notes & Limitations
- This model works only for plant leaf images. If you upload unrelated photos, the prediction may be forced and inaccurate.
- This project is for educational/internship purposes only — accuracy may vary for real-world deployment.
- Current CNN is trained with a basic dataset; for production, it should be upgraded with more images, better data augmentation, and a deeper model.

🛠️ Future Scope
- Use transfer learning with pre-trained models like MobileNet, ResNet for higher accuracy.
- Improve dataset quality and balance for each disease class.
- Add live webcam capture feature in the Streamlit app.
- Deploy as a web service or mobile app.


✨ Author
Name: Prince Timbadiya
Internship: Microsoft AI/ML Virtual Internship 2025
GitHub: PrinceTimbadiya


✅ License
-> This project is open source and free to use for educational purposes.