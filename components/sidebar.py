"""
components/sidebar.py
Renders the sidebar and returns user-controlled settings.
"""

import streamlit as st
from config.settings import BACKBONE_NAME, FUSION_DIM, TARGET_SIZE, CLASS_NAMES
from utils.transforms import get_tta_transforms


def render_sidebar() -> bool:
    """
    Render sidebar widgets and return `use_tta` boolean.
    """
    with st.sidebar:
        st.header("⚙️ Pengaturan")

        if st.button(
            "🗑️ Clear Model Cache",
            help="Gunakan jika model tidak ter-load dengan benar",
        ):
            st.cache_resource.clear()
            st.success("✅ Cache dihapus! Refresh halaman untuk reload model.")

        st.divider()

        use_tta = st.checkbox(
            "Gunakan TTA (Test-Time Augmentation)",
            value=True,
            help=(
                "TTA meningkatkan akurasi dengan merata-rata prediksi "
                "dari beberapa augmentasi berbeda."
            ),
        )

        st.divider()
        _render_model_info(use_tta)

        st.divider()
        _render_disclaimer()

    return use_tta


def _render_model_info(use_tta: bool) -> None:
    st.header("📊 Info Model")
    tta_count = len(get_tta_transforms()) if use_tta else 0
    st.info(
        f"- **Backbone**: {BACKBONE_NAME}\n"
        f"- **Fusion Dim**: {FUSION_DIM}\n"
        f"- **Input Size**: {TARGET_SIZE}×{TARGET_SIZE}\n"
        f"- **Classes**: {', '.join(CLASS_NAMES)}\n"
        f"- **TTA Transforms**: {tta_count}"
    )


def _render_disclaimer() -> None:
    st.header("⚠️ Disclaimer")
    st.warning(
        "Aplikasi ini adalah **demo penelitian** dan **TIDAK** dapat digunakan "
        "sebagai diagnosis medis. Konsultasikan dengan profesional kesehatan "
        "untuk diagnosis yang akurat."
    )
