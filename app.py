from huggingface_hub import hf_hub_download
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import os
import cv2 # Digunakan untuk konversi warna jika diperlukan, tapi dihindari untuk MTCNN detect
from facenet_pytorch import MTCNN

# --- Konfigurasi ---
# Load model dari Hugging Face
MODEL_PATH = hf_hub_download(repo_id="Artz-03/autismeClassification", filename="best_model_phase2_crop_78.pt")
# Nama kelas Anda harus sesuai dengan urutan indeks yang digunakan saat pelatihan
CLASS_NAMES = ['Autistic', 'Non_Autistic']
TARGET_SIZE = 224 # Ukuran gambar yang diharapkan oleh model Anda

@st.cache_resource # Cache resource untuk MTCNN model
def load_mtcnn_model():
    # MTCNN akan berjalan di CPU di Streamlit Cloud
    # image_size di MTCNN adalah untuk output gambar yang di-crop, bukan input deteksi
    return MTCNN(
        image_size=TARGET_SIZE,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False, # Jangan lakukan post-processing otomatis
        device='cpu'
    )

mtcnn = load_mtcnn_model()

def process_image_for_model(img_pil, mtcnn_detector, target_size=224):
    """
    Fungsi untuk mendeteksi dan meng-crop wajah, lalu mempersiapkan gambar untuk model.
    Args:
        img_pil (PIL.Image.Image): Gambar input dalam format PIL Image.
        mtcnn_detector (MTCNN): Objek MTCNN yang sudah diinisialisasi.
        target_size (int): Ukuran target untuk gambar yang diproses.
    Returns:
        torch.Tensor: Tensor gambar yang sudah diproses, siap untuk model.
    """
    # MTCNN dapat langsung menerima PIL Image atau NumPy array (RGB).
    # Memberikan PIL Image secara langsung adalah cara yang paling bersih.
    boxes, _ = mtcnn_detector.detect(img_pil)

    processed_pil_image = None # Akan menyimpan PIL Image dari wajah yang di-crop atau gambar asli

    if boxes is not None and len(boxes) > 0:
        # Pilih wajah terbesar atau wajah pertama
        best_box = boxes[0]
        if len(boxes) > 1:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            best_box = boxes[np.argmax(areas)]

        x1, y1, x2, y2 = [int(b) for b in best_box]

        # Pastikan koordinat dalam batas gambar asli
        width, height = img_pil.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Lakukan cropping dari PIL Image asli
        if x1 >= x2 or y1 >= y2: # Periksa area crop yang tidak valid (lebar/tinggi nol atau negatif)
            st.warning("Koordinat wajah yang terdeteksi tidak valid (area crop kosong). Menggunakan gambar asli.")
            processed_pil_image = img_pil
        else:
            cropped_face_pil = img_pil.crop((x1, y1, x2, y2))
            # Periksa apakah hasil crop kosong (misalnya jika koordinat sangat kecil)
            if cropped_face_pil.size[0] == 0 or cropped_face_pil.size[1] == 0:
                st.warning("Wajah terdeteksi, tetapi area crop kosong setelah cropping. Menggunakan gambar asli.")
                processed_pil_image = img_pil
            else:
                processed_pil_image = cropped_face_pil
    else:
        st.warning("Tidak ada wajah terdeteksi. Menggunakan gambar asli untuk prediksi.")
        processed_pil_image = img_pil # Jika tidak ada wajah, gunakan seluruh gambar

    # Transformasi yang digunakan saat pelatihan
    # Karena 'processed_pil_image' sudah berupa PIL Image, 'transforms.ToPILImage()' tidak diperlukan.
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)), # Mengubah ukuran gambar PIL
        transforms.ToTensor(),                          # Mengubah PIL Image menjadi PyTorch Tensor (HWC -> CHW, 0-255 -> 0.0-1.0)
        transforms.Normalize([0.5]*3, [0.5]*3)          # Normalisasi Tensor
    ])

    return transform(processed_pil_image)

# --- Muat Model PyTorch yang Telah Dilatih ---
# Gunakan st.cache_resource untuk melakukan caching model
@st.cache_resource
def load_model():
    model = timm.create_model("mobilevit_s", pretrained=False, num_classes=len(CLASS_NAMES))
    # Muat state_dict di CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval() # Set model ke mode evaluasi
    return model

model = load_model()

# --- Aplikasi Streamlit ---
st.title("Klasifikasi Autisme pada Wajah")
st.write("Unggah gambar wajah untuk memprediksi apakah orang tersebut autisme atau non-autisme.")

uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

    # Pre-process gambar dan lakukan prediksi
    if st.button("Prediksi"):
        with st.spinner('Memproses dan memprediksi...'):
            try:
                # Proses gambar: deteksi/crop wajah dan transformasi
                input_tensor = process_image_for_model(image, mtcnn, TARGET_SIZE)
                input_batch = input_tensor.unsqueeze(0) # Tambahkan dimensi batch (batch_size=1)

                # Lakukan inferensi
                with torch.no_grad(): # Nonaktifkan perhitungan gradien untuk inferensi
                    output = model(input_batch)
                    probabilities = torch.softmax(output, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][prediction].item()

                predicted_class = CLASS_NAMES[prediction]

                if predicted_class == 'Autistic':
                    st.error(f"Prediksi: **{predicted_class}**")
                else:
                    st.success(f"Prediksi: **{predicted_class}**")
                    
                st.write(f"Keyakinan: **{confidence:.2f}**")

                # --- Bagian untuk menampilkan gambar wajah yang sudah di-crop ---
                st.subheader("Gambar Wajah yang Digunakan untuk Prediksi:")
                # Ulangi proses cropping hanya untuk tampilan. Gunakan PIL Image asli.
                boxes_display, _ = mtcnn.detect(image) # Langsung berikan PIL Image untuk deteksi

                if boxes_display is not None and len(boxes_display) > 0:
                    best_box_display = boxes_display[0]
                    if len(boxes_display) > 1:
                        areas_display = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes_display]
                        best_box_display = boxes_display[np.argmax(areas_display)]

                    x1, y1, x2, y2 = [int(b) for b in best_box_display]

                    # Pastikan koordinat dalam batas gambar asli
                    width_display, height_display = image.size
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width_display, x2)
                    y2 = min(height_display, y2)

                    if x1 >= x2 or y1 >= y2: # Periksa area crop yang tidak valid
                        st.image(image, caption="Tidak dapat meng-crop wajah. Menampilkan gambar asli.", use_column_width=True)
                    else:
                        cropped_face_display_pil = image.crop((x1, y1, x2, y2))
                        if cropped_face_display_pil.size[0] == 0 or cropped_face_display_pil.size[1] == 0:
                            st.image(image, caption="Tidak dapat meng-crop wajah. Menampilkan gambar asli.", use_column_width=True)
                        else:
                            st.image(cropped_face_display_pil, caption="Wajah yang di-crop", use_column_width=True)
                else:
                    st.image(image, caption="Tidak ada wajah terdeteksi. Menampilkan gambar asli.", use_column_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
                st.error("Pastikan gambar adalah gambar wajah yang jelas dan model 'best_model_phase2_nocrop.pt' ada di direktori yang sama.")

st.sidebar.header("Tentang Aplikasi Ini")
st.sidebar.info(
    "Aplikasi ini menggunakan model deep learning MobileViT yang telah dilatih "
    "untuk mengklasifikasikan gambar wajah sebagai 'Autisme' atau 'Non-Autisme'. "
    "Model ini sebelumnya telah melalui tahap *pre-processing* dengan MTCNN untuk "
    "mendeteksi dan meng-crop wajah. "
    "**Catatan**: Ini adalah aplikasi demo dan hasil prediksi tidak boleh digunakan "
    "sebagai diagnosis medis."
)
