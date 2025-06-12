from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention

# Custom Transformer Layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):  # Tambahkan **kwargs
        super(TransformerBlock, self).__init__(**kwargs)  # Oper juga ke super
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        inputs_expanded = tf.expand_dims(inputs, axis=1)
        attn_output = self.att(inputs_expanded, inputs_expanded, training=training)
        attn_output = tf.squeeze(attn_output, axis=1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def get_config(self):
    config = super().get_config()
    config.update({
        "embed_dim": self.att.key_dim,
        "num_heads": self.att.num_heads,
        "ff_dim": self.ffn.layers[0].units,
        "rate": self.dropout1.rate,
    })
    return config

from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

model_path = hf_hub_download(repo_id="Artz-03/autismeClassification", filename="autism_hybrid_model.h5")
model = load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

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
