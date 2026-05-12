"""
Autism Classification App — Entry Point
Run with: streamlit run app.py
"""

import streamlit as st
from config.settings import PAGE_CONFIG
from components.sidebar import render_sidebar
from components.tabs import render_main_tabs

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(**PAGE_CONFIG)

# ── Load custom CSS ──────────────────────────────────────────────────────────
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🧠 Klasifikasi Autisme dari Gambar Wajah")
st.markdown(
    "Model **MobileViTV2** dengan arsitektur *Fusion Backbone Classifier* "
    "mengklasifikasikan gambar wajah sebagai **Autistic** atau **Non-Autistic**."
)

# ── Sidebar → returns user settings ─────────────────────────────────────────
use_tta = render_sidebar()

# ── Main content ─────────────────────────────────────────────────────────────
render_main_tabs(use_tta)
