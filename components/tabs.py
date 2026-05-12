"""
components/tabs.py
Renders the two main input tabs: file upload and live camera.
"""

import streamlit as st
from PIL import Image

from components.prediction_result import render_prediction
from components.instructions import render_upload_instructions, render_camera_instructions


def render_main_tabs(use_tta: bool) -> None:
    tab_upload, tab_camera = st.tabs(["📁 Upload Gambar", "📷 Ambil Foto dari Kamera"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Unggah Gambar Wajah",
            type=["jpg", "jpeg", "png"],
            help="Unggah foto wajah yang jelas untuk hasil terbaik",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            render_prediction(image, use_tta)
        else:
            render_upload_instructions()

    with tab_camera:
        render_camera_instructions()
        camera_photo = st.camera_input("Ambil Foto Langsung")

        if camera_photo is not None:
            image = Image.open(camera_photo).convert("RGB")
            render_prediction(image, use_tta)
