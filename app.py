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
BACKBONE_NAME = "mobilevitv2_100"  # Sesuai dengan training

# ImageNet normalization (SESUAI TRAINING!)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Download model dari Hugging Face
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
def load_model():
    model = FusionBackboneClassifier(
        backbone_name=BACKBONE_NAME,
        out_indices=(1, 2, 3),
        fusion_dim=768,  # ✅ Sudah benar
        num_classes=len(CLASS_NAMES),  # ✅ Sudah 2 class
        fusion_dropout=0.4,
        classifier_dropout=0.25,
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # Ambil state_dict dari berbagai format
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Hilangkan prefix module. (DataParallel)
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    # LOAD STATE DICT
    model.load_state_dict(state_dict, strict=True)
    
    # HAPUS BAGIAN REBUILD CLASSIFIER - TIDAK PERLU!
    # Classifier sudah ada di checkpoint dan sudah sesuai dengan 2 class

    model.eval()
    return model

mtcnn = load_mtcnn_model()
model = load_model()


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def get_transforms():
    """Transform yang SAMA dengan validation/test di training"""
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms():
    """TTA transforms untuk meningkatkan akurasi"""
    return [
        # Original
        A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        # Horizontal Flip
        A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        # Scale Up
        A.Compose([
            A.Resize(300, 300),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        # Slight Rotation
        A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Rotate(limit=(10, 10), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
    ]


def process_image_for_model(img_pil, mtcnn_detector, target_size=224):
    """
    Deteksi dan crop wajah, lalu persiapkan untuk model
    """
    boxes, _ = mtcnn_detector.detect(img_pil)
    processed_pil_image = None

    if boxes is not None and len(boxes) > 0:
        # Pilih wajah terbesar
        best_box = boxes[0]
        if len(boxes) > 1:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            best_box = boxes[np.argmax(areas)]

        x1, y1, x2, y2 = [int(b) for b in best_box]
        width, height = img_pil.size
        
        # Clamp coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x1 >= x2 or y1 >= y2:
            st.warning("⚠️ Koordinat wajah tidak valid. Menggunakan gambar asli.")
            processed_pil_image = img_pil
        else:
            cropped_face_pil = img_pil.crop((x1, y1, x2, y2))
            if cropped_face_pil.size[0] == 0 or cropped_face_pil.size[1] == 0:
                st.warning("⚠️ Area crop kosong. Menggunakan gambar asli.")
                processed_pil_image = img_pil
            else:
                processed_pil_image = cropped_face_pil
    else:
        st.warning("⚠️ Tidak ada wajah terdeteksi. Menggunakan gambar asli.")
        processed_pil_image = img_pil

    return processed_pil_image


def predict_single(image_np, model, transform):
    """Prediksi tunggal tanpa TTA"""
    transformed = transform(image=image_np)['image']
    input_batch = transformed.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence, probabilities[0].numpy()


def predict_with_tta(image_np, model, tta_transforms):
    """Prediksi dengan TTA untuk akurasi lebih tinggi"""
    all_probs = []
    
    with torch.no_grad():
        for i, transform in enumerate(tta_transforms):
            transformed = transform(image=image_np)['image']
            input_batch = transformed.unsqueeze(0)
            
            output = model(input_batch)
            probs = torch.softmax(output, dim=1).cpu().numpy()
            
            # Beri bobot lebih pada original image
            weight = 1.5 if i == 0 else 1.0
            all_probs.append(probs * weight)
    
    # Weighted average
    total_weight = 1.5 + (len(tta_transforms) - 1) * 1.0
    avg_probs = np.sum(all_probs, axis=0) / total_weight
    
    prediction = np.argmax(avg_probs[0])
    confidence = avg_probs[0][prediction]
    
    return prediction, confidence, avg_probs[0]


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Autism Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Klasifikasi Autisme dari Gambar Wajah")
st.markdown("""
Aplikasi ini menggunakan deep learning model **MobileViTV2** dengan arsitektur *Fusion Backbone Classifier* 
untuk mengklasifikasikan gambar wajah sebagai **Autistic** atau **Non-Autistic**.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    use_tta = st.checkbox(
        "Gunakan TTA (Test-Time Augmentation)", 
        value=True,
        help="TTA meningkatkan akurasi dengan rata-rata prediksi dari beberapa augmentasi"
    )
    
    st.markdown("---")
    st.header("📊 Info Model")
    st.info(f"""
    - **Backbone**: {BACKBONE_NAME}
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

# Main content
uploaded_file = st.file_uploader(
    "📤 Unggah Gambar Wajah", 
    type=["jpg", "jpeg", "png"],
    help="Unggah foto wajah yang jelas untuk hasil terbaik"
)

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Gambar Asli")
        st.image(image, use_column_width=True)
    
    # Tombol prediksi
    if st.button("🔮 Mulai Prediksi", type="primary"):
        with st.spinner('🔄 Memproses gambar...'):
            try:
                # 1. Face detection & cropping
                processed_pil = process_image_for_model(image, mtcnn, TARGET_SIZE)
                processed_np = np.array(processed_pil)
                
                with col2:
                    st.subheader("✂️ Wajah yang Dideteksi")
                    st.image(processed_pil, use_column_width=True)
                
                # 2. Prediction
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
                
                # 3. Display results
                st.markdown("---")
                predicted_class = CLASS_NAMES[prediction]
                
                # Result card
                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                
                with result_col2:
                    if predicted_class == 'Autistic':
                        st.error(f"### 🔴 Prediksi: **{predicted_class}**")
                    else:
                        st.success(f"### 🟢 Prediksi: **{predicted_class}**")
                    
                    # Confidence bar
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    st.progress(confidence)
                    
                    # Probabilitas untuk setiap kelas
                    st.markdown("#### 📊 Probabilitas per Kelas:")
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob_pct = probs[i] * 100
                        st.write(f"**{class_name}**: {prob_pct:.2f}%")
                        st.progress(probs[i])
                
                # Interpretasi
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

else:
    # Instruksi awal
    st.info("""
    👆 **Cara Menggunakan:**
    1. Unggah foto wajah (JPG/PNG)
    2. Aktifkan/nonaktifkan TTA di sidebar (opsional)
    3. Klik tombol "Mulai Prediksi"
    4. Lihat hasil klasifikasi
    """)
    
    # Demo image placeholder
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
