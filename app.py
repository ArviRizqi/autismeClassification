from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, Dense, Dropout, LayerNormalization
import streamlit as st
import numpy as np
from PIL import Image
from model_config import TransformerBlock  # Impor custom layer

custom_objects = {
    'TransformerBlock': TransformerBlock,
    'MultiHeadAttention': MultiHeadAttention,
    'Dense': Dense,
    'Dropout': Dropout,
    'LayerNormalization': LayerNormalization,
}

# Load model dari Hugging Face
model_path = hf_hub_download(repo_id="Artz-03/autismeClassification", filename="hybrid_autism_model.keras")

try:
    model = load_model(model_path, custom_objects=custom_objects)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("🧠 Deteksi Autisme dari Gambar Anak")

uploaded_file = st.file_uploader("Upload gambar wajah anak", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    if prediction <= 0.5:
        st.error(f"Model memprediksi: **Autistik** (probabilitas: {1 - prediction:.2f})")
    else:
        st.success(f"Model memprediksi: **Tidak Autistik** (probabilitas: {prediction:.2f})")



