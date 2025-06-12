import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Title
st.title("Autism Detection from Image")
st.markdown("Upload an image to predict whether it's classified as **Autistic** or **Non-Autistic**.")

# Load the model
@st.cache_resource
def load_hybrid_model():
    model = load_model('autism_cnn_transformer_model.h5', compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_hybrid_model()

# Image Preprocessing
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Batch dimension
    return img

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0][0]

    # Display result
    if prediction > 0.5:
        st.error(f"Prediction: Autistic ({prediction:.2f})")
    else:
        st.success(f"Prediction: Non-Autistic ({1 - prediction:.2f})")
