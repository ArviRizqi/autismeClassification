import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Unduh model dari Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="Artz-03/autismeClassification", filename="autism_hybrid_model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Preprocess function
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# UI
st.title("Deteksi Autisme dari Gambar Anak")

uploaded_file = st.file_uploader("Upload gambar wajah anak", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error(f"Model memprediksi: **Autistik** (probabilitas: {prediction:.2f})")
    else:
        st.success(f"Model memprediksi: **Tidak Autistik** (probabilitas: {1 - prediction:.2f})")
