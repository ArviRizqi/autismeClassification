from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
import streamlit as st

# Load model dari Hugging Face
model_path = hf_hub_download(repo_id="Artz-03/autismeClassification", filename="model_fine_tuned_87.h5")
model = load_model(model_path)

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Autisme", layout="centered")

st.title("🧠 Deteksi Autisme dari Gambar")
st.write("Upload gambar wajah untuk memprediksi apakah termasuk kategori autis atau tidak.")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Resize dan ubah jadi array
    img = img.resize((224, 224))  # ubah sesuai input model Anda
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    pred = model.predict(img_array)[0][0]
    label = "Autis 🧩" if pred > 0.5 else "Non-Autis 🙂"

    st.write(f"Hasil Prediksi: **{label}**")
    st.write(f"Confidence Score: {pred:.4f}")
