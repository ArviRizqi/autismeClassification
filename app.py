import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Autisme", layout="centered")

st.title("🧠 Deteksi Autisme dari Gambar")
st.write("Upload gambar wajah untuk memprediksi apakah termasuk kategori autis atau tidak.")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocess
    img_size = (224, 224)
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    prediction = model.predict(img_array)[0][0]

    # Tampilkan hasil
    if prediction > 0.5:
        st.error("Hasil: Autis 🧩")
    else:
        st.success("Hasil: Non-Autis 🙂")

    st.write(f"Confidence: {prediction:.4f}")
