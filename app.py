# app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from huggingface_hub import hf_hub_download

from model_config import TransformerBlock  # Layer kustom

# Konfigurasi custom_objects
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'Dense': Dense,
    'Dropout': Dropout,
    'LayerNormalization': LayerNormalization,
    'MultiHeadAttention': MultiHeadAttention
}

# Load model dari HuggingFace
model_path = hf_hub_download(repo_id="Artz-03/autismeClassification", filename="autisme-classifier.keras")

st.title("🧠 Deteksi Autisme dari Gambar Wajah Anak")

try:
    model = load_model(model_path, custom_objects=custom_objects)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")

def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

uploaded_file = st.file_uploader("Upload gambar wajah anak", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    if prediction <= 0.5:
        st.error(f"🚨 Model memprediksi: **Autistik** (probabilitas: {1 - prediction:.2f})")
    else:
        st.success(f"✅ Model memprediksi: **Tidak Autistik** (probabilitas: {prediction:.2f})")
