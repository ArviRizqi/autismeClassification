"""
components/instructions.py
Static helper panels: how-to guide and image quality tips.
"""

import streamlit as st


def render_upload_instructions() -> None:
    st.info(
        "👆 **Cara Menggunakan:**\n"
        "1. Unggah foto wajah (JPG/PNG)\n"
        "2. Aktifkan/nonaktifkan TTA di sidebar (opsional)\n"
        "3. Klik tombol **Mulai Prediksi**\n"
        "4. Lihat hasil klasifikasi"
    )
    st.divider()
    _render_quality_tips()


def render_camera_instructions() -> None:
    st.info(
        "📷 **Cara Menggunakan Kamera:**\n"
        "1. Klik tombol kamera di bawah\n"
        "2. Izinkan akses kamera di browser\n"
        "3. Arahkan kamera ke wajah\n"
        "4. Klik **Take Photo**\n"
        "5. Klik tombol **Mulai Prediksi**"
    )


def _render_quality_tips() -> None:
    st.subheader("📝 Tips untuk Hasil Terbaik:")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("✅ **Good**")
        st.markdown(
            "- Wajah terlihat jelas\n"
            "- Pencahayaan baik\n"
            "- Resolusi tinggi\n"
            "- Satu wajah dominan"
        )
    with col2:
        st.warning("⚠️ **Acceptable**")
        st.markdown(
            "- Wajah agak miring\n"
            "- Pencahayaan normal\n"
            "- Resolusi sedang\n"
            "- Beberapa wajah"
        )
    with col3:
        st.error("❌ **Avoid**")
        st.markdown(
            "- Wajah tertutup\n"
            "- Terlalu gelap/terang\n"
            "- Resolusi rendah\n"
            "- Tidak ada wajah"
        )
