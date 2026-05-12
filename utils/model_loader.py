"""
utils/model_loader.py
Loads MTCNN face detector and FusionBackboneClassifier from HuggingFace Hub.
Both are cached with @st.cache_resource so they are only loaded once.
"""

import torch
import streamlit as st
from huggingface_hub import hf_hub_download
from facenet_pytorch import MTCNN

from config.settings import (
    BACKBONE_NAME, OUT_INDICES, FUSION_DIM, NUM_CLASSES,
    FUSION_DROPOUT, CLASSIFIER_DROPOUT,
    HF_REPO_ID, HF_FILENAME,
    CLASS_NAMES, MTCNN_CONFIG,
)
from utils.model_architecture import FusionBackboneClassifier


@st.cache_resource(show_spinner="🔍 Memuat face detector…")
def load_mtcnn() -> MTCNN:
    return MTCNN(**MTCNN_CONFIG)


@st.cache_resource(show_spinner="🧠 Memuat model klasifikasi…")
def load_classifier() -> FusionBackboneClassifier:
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)

    model = FusionBackboneClassifier(
        backbone_name      = BACKBONE_NAME,
        out_indices        = OUT_INDICES,
        fusion_dim         = FUSION_DIM,
        num_classes        = NUM_CLASSES,
        fusion_dropout     = FUSION_DROPOUT,
        classifier_dropout = CLASSIFIER_DROPOUT,
    )

    checkpoint = torch.load(model_path, map_location="cpu")

    # Support various checkpoint formats
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    # Strip DataParallel "module." prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        st.warning(f"⚠️ strict load failed — falling back to strict=False.\n{exc}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model
