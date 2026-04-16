from huggingface_hub import hf_hub_download
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================================
# STREAMLIT PAGE CONFIG - HARUS PALING PERTAMA!
# ============================================================================
st.set_page_config(
    page_title="Autism Classification",
    page_icon="🧠",
    layout="wide"
)

# ============================================================================
# DEFINISI ARSITEKTUR MODEL (SAMA DENGAN TRAINING)
# ============================================================================

class SELayer(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OptimizedFeatureFusionBlock(nn.Module):
    def __init__(self, input_channels, output_channels=512, dropout_rate=0.4):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.SiLU(inplace=True)
            ) for c in input_channels
        ])

        self.gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_channels * len(input_channels), output_channels, 1),
            nn.Sigmoid()
        )

        self.se_block = SELayer(output_channels, reduction=8)
        self.fuse_conv = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(output_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feature_maps):
        target_h, target_w = feature_maps[-1].shape[2:]
        projected = []
        for fmap, proj in zip(feature_maps, self.projs):
            p = F.interpolate(fmap, size=(target_h, target_w), mode='bilinear', align_corners=False)
            projected.append(proj(p))

        cat_feats = torch.cat(projected, dim=1)
        gate = self.gate_gen(cat_feats)
        fused = sum(projected) * gate

        refined = self.fuse_bn(self.fuse_conv(fused))
        fused = F.silu(refined + fused)
        fused = self.se_block(fused)

        gap = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(fused, 1).flatten(1)
        return self.dropout(gap + gmp)


class FusionBackboneClassifier(nn.Module):
    def __init__(self, backbone_name="mobilevitv2_100", out_indices=(1,2,3),
                 fusion_dim=768, num_classes=2,
                 fusion_dropout=0.4, classifier_dropout=0.25):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, features_only=True, out_indices=out_indices
        )
        in_chs = self.backbone.feature_info.channels()

        self.fusion = OptimizedFeatureFusionBlock(
            in_chs, output_channels=fusion_dim, dropout_rate=fusion_dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        fused_vec = self.fusion(feats)
        logits = self.classifier(fused_vec)
        return logits


# ============================================================================
# KONFIGURASI
# ============================================================================

CLASS_NAMES = ['Autistic', 'Non_Autistic']
TARGET_SIZE = 224
BACKBONE_NAME = "mobilevitv2_100"
FUSION_DIM = 768

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_PATH = hf_hub_download(
    repo_id="Artz-03/autismeClassification",
    filename="mobilevitv2_phase2_optimized.pth"
)


# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_mtcnn_model():
    return MTCNN(
        image_size=TARGET_SIZE,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device='cpu'
    )


@st.cache_resource
def load_model_v2():
    model = FusionBackboneClassifier(
        backbone_name=BACKBONE_NAME,
        out_indices=(1, 2, 3),
        fusion_dim=FUSION_DIM,
        num_classes=len(CLASS_NAMES),
        fusion_dropout=0.4,
        classifier_dropout=0.25,
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully dengan strict=True")
    except RuntimeError as e:
        print(f"⚠️ Error loading dengan strict=True: {e}")
        print("🔄 Mencoba load dengan strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


mtcnn = load_mtcnn_model()
model = load_model_v2()


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms():
    return [
        A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(300, 300),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Rotate(limit=10, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
    ]


def process_image_for_model(img_pil, mtcnn_detector, target_size=224):
    boxes, _ = mtcnn_detector.detect(img_pil)

    if boxes is not None and len(boxes) > 0:
        if len(boxes) > 1:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            best_box = boxes[np.argmax(areas)]
        else:
            best_box = boxes[0]

        x1, y1, x2, y2 = [int(b) for b in best_box]
        width, height = img_pil.size

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x1 < x2 and y1 < y2:
            cropped_face = img_pil.crop((x1, y1, x2, y2))
            if cropped_face.size[0] > 0 and cropped_face.size[1] > 0:
                return cropped_face

    st.warning("⚠️ Wajah tidak terdeteksi dengan jelas. Menggunakan gambar asli.")
    return img_pil


def predict_single(image_np, model, transform):
    transformed = transform(image=image_np)['image']
    input_batch = transformed.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    probs_array = probabilities[0].cpu().numpy().astype(float)
    return prediction, float(confidence), probs_array


def predict_with_tta(image_np, model, tta_transforms):
    all_probs = []

    with torch.no_grad():
        for i, transform in enumerate(tta_transforms):
            transformed = transform(image=image_np)['image']
            input_batch = transformed.unsqueeze(0)

            output = model(input_batch)
            probs = torch.softmax(output, dim=1).cpu().numpy()

            weight = 1.5 if i == 0 else 1.0
            all_probs.append(probs * weight)

    total_weight = 1.5 + (len(tta_transforms) - 1) * 1.0
    avg_probs = np.sum(all_probs, axis=0) / total_weight

    prediction = int(np.argmax(avg_probs[0]))
    confidence = float(avg_probs[0][prediction])
    probs_array = avg_probs[0].astype(float)

    return prediction, confidence, probs_array


# ============================================================================
# HELPER: TAMPILKAN HASIL PREDIKSI
# ============================================================================

def show_results(image, use_tta):
    """Proses gambar dan tampilkan hasil prediksi"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Gambar Input")
        st.image(image, use_column_width=True)

    if st.button("🔮 Mulai Prediksi", type="primary"):
        with st.spinner('🔄 Memproses gambar...'):
            try:
                processed_pil = process_image_for_model(image, mtcnn, TARGET_SIZE)
                processed_np = np.array(processed_pil)

                with col2:
                    st.subheader("✂️ Wajah yang Diproses")
                    st.image(processed_pil, use_column_width=True)

                if use_tta:
                    st.info("🔄 Menggunakan TTA (4 augmentations)...")
                    tta_transforms = get_tta_transforms()
                    prediction, confidence, probs = predict_with_tta(
                        processed_np, model, tta_transforms
                    )
                else:
                    st.info("⚡ Prediksi cepat (tanpa TTA)...")
                    transform = get_transforms()
                    prediction, confidence, probs = predict_single(
                        processed_np, model, transform
                    )

                st.markdown("---")
                predicted_class = CLASS_NAMES[prediction]

                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

                with result_col2:
                    if predicted_class == 'Autistic':
                        st.error(f"### 🔴 Prediksi: **{predicted_class}**")
                    else:
                        st.success(f"### 🟢 Prediksi: **{predicted_class}**")

                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    conf_value = min(max(confidence, 0.0), 1.0)
                    st.progress(conf_value)

                    st.markdown("#### 📊 Probabilitas per Kelas:")
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob_pct = probs[i] * 100
                        st.write(f"**{class_name}**: {prob_pct:.2f}%")
                        prob_value = min(max(probs[i], 0.0), 1.0)
                        st.progress(prob_value)

                st.markdown("---")
                st.subheader("💡 Interpretasi Hasil")

                if confidence > 0.9:
                    conf_level = "sangat tinggi"
                    emoji = "🎯"
                elif confidence > 0.75:
                    conf_level = "tinggi"
                    emoji = "✅"
                elif confidence > 0.6:
                    conf_level = "sedang"
                    emoji = "⚠️"
                else:
                    conf_level = "rendah"
                    emoji = "❓"

                st.info(f"""
                {emoji} Model memiliki **confidence {conf_level}** ({confidence*100:.1f}%)
                bahwa gambar ini termasuk kelas **{predicted_class}**.

                {'**TTA digunakan**: Prediksi ini adalah hasil rata-rata dari 4 augmentasi berbeda untuk akurasi lebih tinggi.' if use_tta else '**TTA tidak digunakan**: Untuk akurasi lebih tinggi, aktifkan TTA di sidebar.'}
                """)

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
                st.exception(e)


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🧠 Klasifikasi Autisme dari Gambar Wajah")
st.markdown("""
Aplikasi ini menggunakan deep learning model **MobileViTV2** dengan arsitektur *Fusion Backbone Classifier*
untuk mengklasifikasikan gambar wajah sebagai **Autistic** atau **Non-Autistic**.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")

    if st.button("🗑️ Clear Model Cache", help="Gunakan jika model tidak ter-load dengan benar"):
        st.cache_resource.clear()
        st.success("✅ Cache dihapus! Refresh halaman untuk reload model.")

    st.markdown("---")

    use_tta = st.checkbox(
        "Gunakan TTA (Test-Time Augmentation)",
        value=True,
        help="TTA meningkatkan akurasi dengan rata-rata prediksi dari beberapa augmentasi"
    )

    st.markdown("---")
    st.header("📊 Info Model")
    st.info(f"""
    - **Backbone**: {BACKBONE_NAME}
    - **Fusion Dim**: {FUSION_DIM}
    - **Input Size**: {TARGET_SIZE}x{TARGET_SIZE}
    - **Classes**: {', '.join(CLASS_NAMES)}
    - **TTA Transforms**: {len(get_tta_transforms()) if use_tta else 0}
    """)

    st.markdown("---")
    st.header("⚠️ Disclaimer")
    st.warning("""
    Aplikasi ini adalah **demo penelitian** dan **TIDAK** dapat digunakan
    sebagai diagnosis medis. Konsultasikan dengan profesional kesehatan
    untuk diagnosis yang akurat.
    """)

# ============================================================================
# INPUT GAMBAR: TAB UPLOAD & KAMERA
# ============================================================================

tab_upload, tab_camera = st.tabs(["📁 Upload Gambar", "📷 Ambil Foto dari Kamera"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Unggah Gambar Wajah",
        type=["jpg", "jpeg", "png"],
        help="Unggah foto wajah yang jelas untuk hasil terbaik"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        show_results(image, use_tta)
    else:
        st.info("""
        👆 **Cara Menggunakan:**
        1. Unggah foto wajah (JPG/PNG)
        2. Aktifkan/nonaktifkan TTA di sidebar (opsional)
        3. Klik tombol "Mulai Prediksi"
        4. Lihat hasil klasifikasi
        """)

        st.markdown("---")
        st.subheader("📝 Tips untuk Hasil Terbaik:")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.success("✅ **Good**")
            st.markdown("""
            - Wajah terlihat jelas
            - Pencahayaan baik
            - Resolusi tinggi
            - Satu wajah dominan
            """)

        with col2:
            st.warning("⚠️ **Acceptable**")
            st.markdown("""
            - Wajah agak miring
            - Pencahayaan normal
            - Resolusi sedang
            - Beberapa wajah
            """)

        with col3:
            st.error("❌ **Avoid**")
            st.markdown("""
            - Wajah tertutup
            - Terlalu gelap/terang
            - Resolusi rendah
            - Tidak ada wajah
            """)

with tab_camera:
    st.markdown("Pastikan browser mengizinkan akses kamera, lalu arahkan kamera ke wajah dan klik **Take Photo**.")
    camera_photo = st.camera_input("Ambil Foto Langsung")

    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        show_results(image, use_tta)
    else:
        st.info("""
        📷 **Cara Menggunakan Kamera:**
        1. Klik tombol kamera di atas
        2. Izinkan akses kamera di browser
        3. Arahkan kamera ke wajah
        4. Klik **Take Photo**
        5. Klik tombol "Mulai Prediksi"
        """)
