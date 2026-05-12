"""
components/prediction_result.py
Renders the full prediction UI: image preview, results, interpretation.
"""

import numpy as np
import streamlit as st
from PIL import Image

from config.settings import CLASS_NAMES
from utils import (
    load_mtcnn, load_classifier,
    detect_and_crop_face,
    predict_single, predict_with_tta,
    get_standard_transform, get_tta_transforms,
    confidence_label,
)


def render_prediction(source_image: Image.Image, use_tta: bool) -> None:
    """
    Display the uploaded/captured image, run prediction on button click,
    and show results + interpretation.
    """
    col_img, col_face = st.columns(2)

    with col_img:
        st.subheader("📷 Gambar Input")
        st.image(source_image, use_column_width=True)

    if not st.button("🔮 Mulai Prediksi", type="primary"):
        return

    with st.spinner("🔄 Mendeteksi wajah dan memproses gambar…"):
        try:
            mtcnn     = load_mtcnn()
            model     = load_classifier()

            face_img, face_found = detect_and_crop_face(source_image, mtcnn)

            if not face_found:
                st.warning("⚠️ Wajah tidak terdeteksi dengan jelas. Menggunakan gambar asli.")

            with col_face:
                st.subheader("✂️ Wajah yang Diproses")
                st.image(face_img, use_column_width=True)

            face_np = np.array(face_img)

            if use_tta:
                st.info("🔄 Menggunakan TTA (4 augmentations)…")
                prediction, confidence, probs = predict_with_tta(
                    face_np, model, get_tta_transforms()
                )
            else:
                st.info("⚡ Prediksi cepat (tanpa TTA)…")
                prediction, confidence, probs = predict_single(
                    face_np, model, get_standard_transform()
                )

        except Exception as exc:
            st.error(f"❌ Terjadi kesalahan: {exc}")
            st.exception(exc)
            return

    _render_results(prediction, confidence, probs, use_tta)


# ── Private helpers ───────────────────────────────────────────────────────────

def _render_results(
    prediction: int,
    confidence: float,
    probs: np.ndarray,
    use_tta: bool,
) -> None:
    st.divider()
    predicted_class = CLASS_NAMES[prediction]

    _, result_col, _ = st.columns([1, 2, 1])

    with result_col:
        if predicted_class == "Autistic":
            st.error(f"### 🔴 Prediksi: **{predicted_class}**")
        else:
            st.success(f"### 🟢 Prediksi: **{predicted_class}**")

        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(float(np.clip(confidence, 0.0, 1.0)))

        st.markdown("#### 📊 Probabilitas per Kelas:")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"**{name}**: {probs[i] * 100:.2f}%")
            st.progress(float(np.clip(probs[i], 0.0, 1.0)))

    st.divider()
    _render_interpretation(predicted_class, confidence, use_tta)


def _render_interpretation(
    predicted_class: str,
    confidence: float,
    use_tta: bool,
) -> None:
    st.subheader("💡 Interpretasi Hasil")

    conf_level, emoji = confidence_label(confidence)
    tta_note = (
        "**TTA digunakan**: Prediksi ini adalah hasil rata-rata dari 4 augmentasi "
        "berbeda untuk akurasi lebih tinggi."
        if use_tta
        else "**TTA tidak digunakan**: Untuk akurasi lebih tinggi, aktifkan TTA di sidebar."
    )

    st.info(
        f"{emoji} Model memiliki **confidence {conf_level}** ({confidence * 100:.1f}%) "
        f"bahwa gambar ini termasuk kelas **{predicted_class}**.\n\n{tta_note}"
    )
