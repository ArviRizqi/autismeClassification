import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import os
import cv2 # Hanya jika Anda ingin menyertakan MTCNN di aplikasi deploy untuk cropping langsung
from facenet_pytorch import MTCNN # Hanya jika Anda ingin menyertakan MTCNN di aplikasi deploy untuk cropping langsung

# --- Konfigurasi ---
MODEL_PATH = 'best_model_phase2_nocrop.pt' # Pastikan nama file model sesuai
# Nama kelas Anda harus sesuai dengan urutan indeks yang digunakan saat pelatihan
CLASS_NAMES = ['Autistic', 'Non_Autistic'] 
TARGET_SIZE = 224 # Ukuran gambar yang diharapkan oleh model Anda

@st.cache_resource # Cache resource untuk MTCNN model
def load_mtcnn_model():
    # MTCNN akan berjalan di CPU di Streamlit Cloud
    return MTCNN(
        image_size=TARGET_SIZE,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device='cpu' 
    )

mtcnn = load_mtcnn_model()

def process_image_for_model(img, mtcnn_detector, target_size=224):
    """
    Fungsi untuk mendeteksi dan meng-crop wajah, lalu mempersiapkan gambar untuk model.
    """
    img_np = np.array(img)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # MTCNN expects BGR if using OpenCV for read

    boxes, _ = mtcnn_detector.detect(img_rgb) # mtcnn_detector expects RGB if using PIL/Torch directly

    cropped_face = None
    if boxes is not None and len(boxes) > 0:
        # Pilih wajah terbesar atau wajah pertama
        best_box = boxes[0] 
        if len(boxes) > 1:
            # Optionally, select the largest face
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            best_box = boxes[np.argmax(areas)]

        x1, y1, x2, y2 = [int(b) for b in best_box]

        # Pastikan koordinat dalam batas gambar
        h_orig, w_orig, _ = img_rgb.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_orig, x2)
        y2 = min(h_orig, y2)
        
        cropped_face = img_rgb[y1:y2, x1:x2]
        
        if cropped_face.size == 0:
            st.warning("Wajah terdeteksi, tetapi area crop kosong. Menggunakan gambar asli.")
            cropped_face = img_rgb

    else:
        st.warning("Tidak ada wajah terdeteksi. Menggunakan gambar asli untuk prediksi.")
        cropped_face = img_rgb # Jika tidak ada wajah, gunakan seluruh gambar

    # Transformasi yang digunakan saat pelatihan (sesuaikan jika Anda punya augmentasi di val/test)
    transform = transforms.Compose([
        transforms.ToPILImage(), # Konversi ke PIL Image setelah OpenCV processing
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Konversi cropped_face (numpy array BGR) ke PIL Image RGB untuk transformasi
    # Jika cropped_face awalnya PIL Image RGB, tidak perlu konversi BGR ke RGB
    if isinstance(cropped_face, np.ndarray):
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        cropped_face = Image.fromarray(cropped_face)

    return transform(cropped_face)

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
                input_batch = input_tensor.unsqueeze(0) # Tambahkan dimensi batch

                # Lakukan inferensi
                with torch.no_grad():
                    output = model(input_batch)
                    probabilities = torch.softmax(output, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][prediction].item()

                predicted_class = CLASS_NAMES[prediction]

                st.success(f"Prediksi: **{predicted_class}**")
                st.write(f"Keyakinan: **{confidence:.2f}**")

                # Tambahan: Tampilkan gambar wajah yang sudah di-crop (jika ada)
                # Untuk menampilkan gambar yang di-crop, kita perlu mengulang proses cropping
                # atau mengembalikan gambar yang di-crop dari process_image_for_model.
                # Untuk kesederhanaan, kita bisa membuat fungsi terpisah untuk menampilkan saja.
                st.subheader("Gambar Wajah yang Digunakan untuk Prediksi:")
                # Ulangi proses cropping hanya untuk tampilan
                img_np_display = np.array(image)
                img_rgb_display = cv2.cvtColor(img_np_display, cv2.COLOR_RGB2BGR)
                boxes_display, _ = mtcnn.detect(img_rgb_display)

                if boxes_display is not None and len(boxes_display) > 0:
                    best_box_display = boxes_display[0]
                    if len(boxes_display) > 1:
                        areas_display = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes_display]
                        best_box_display = boxes_display[np.argmax(areas_display)]
                    
                    x1, y1, x2, y2 = [int(b) for b in best_box_display]
                    cropped_face_display = img_rgb_display[max(0, y1):min(img_rgb_display.shape[0], y2), max(0, x1):min(img_rgb_display.shape[1], x2)]
                    
                    if cropped_face_display.size == 0:
                        st.image(image, caption="Tidak dapat meng-crop wajah. Menampilkan gambar asli.", use_column_width=True)
                    else:
                        st.image(cv2.cvtColor(cropped_face_display, cv2.COLOR_BGR2RGB), caption="Wajah yang di-crop", use_column_width=True)
                else:
                    st.image(image, caption="Tidak ada wajah terdeteksi. Menampilkan gambar asli.", use_column_width=True)


            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
                st.error("Pastikan gambar adalah gambar wajah yang jelas.")

st.sidebar.header("Tentang Aplikasi Ini")
st.sidebar.info(
    "Aplikasi ini menggunakan model deep learning MobileViT yang telah dilatih "
    "untuk mengklasifikasikan gambar wajah sebagai 'Autisme' atau 'Non-Autisme'. "
    "Model ini sebelumnya telah melalui tahap *pre-processing* dengan MTCNN untuk "
    "mendeteksi dan meng-crop wajah. "
    "**Catatan**: Ini adalah aplikasi demo dan hasil prediksi tidak boleh digunakan "
    "sebagai diagnosis medis."
)
