import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time

# Load the trained model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")  # Ensure the correct path
    return model

model = load_model()

# Class labels based on training
CLASS_NAMES = ['Healthy', 'Early Blight', 'Late Blight']

# Define Prediction Function (Moved it Up)
def predict(image):
    img_array = np.array(image)

    # Ensure 3 channels
    if img_array.shape[-1] == 4:  # If image has an alpha channel
        img_array = img_array[:, :, :3]

    img_resized = cv2.resize(img_array, (128, 128))  # Resize to match model input
    img_resized = img_resized / 255.0  # Normalize
    img_expanded = np.expand_dims(img_resized, axis=0)  # Expand dimensions for model

    predictions = model.predict(img_expanded)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return f"{predicted_class} ({confidence:.2f}% Confidence)", confidence

# Sidebar Info
st.sidebar.title("ℹ About the Model")
st.sidebar.write("""
- Trained using CNN 🧠
- Image Size: 128x128 px
- 3 Classes: Healthy, Early Blight, Late Blight 🍃
""")
st.sidebar.write("💡 Upload a potato leaf image to detect its disease status.")

# Main UI
st.title("🥔 Potato Leaf Disease Detector")
st.write("Upload an image of a potato leaf to check for diseases using AI.")

# Upload Image
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "png", "jpeg"])

# Camera Option
st.write("📷 Or take a live photo:")
camera_img = st.camera_input("Capture Image")

# Select image source
image_source = None
if uploaded_file is not None:
    image_source = Image.open(uploaded_file)
elif camera_img is not None:
    image_source = Image.open(camera_img)

# Display Image & Predict
if image_source is not None:
    st.image(image_source, caption="📌 Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Disease 🩺"):
        with st.spinner("Analyzing the image... ⏳"):
            time.sleep(2)  # Simulate processing time
            result, confidence = predict(image_source)
            st.success(f"✅ Prediction: {result}")
            
            # Confidence Progress Bar
            st.progress(int(confidence))

if __name__ == "main":
    st.markdown("<style>div.stButton > button {width: 100%;}</style>", unsafe_allow_html=True)

file_id= "1_KQjJRcp_CivsUhdoOWASQlYL25QzZA2"
url = 'https://drive.google.com/file/d/1_KQjJRcp_CivsUhdoOWASQlYL25QzZA2/view?usp=sharing'
model_path="trained_plant_disease_model.keras"