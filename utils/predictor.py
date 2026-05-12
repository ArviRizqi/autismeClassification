"""
utils/predictor.py
Face detection and model inference helpers.
All functions are pure (no Streamlit calls) — UI concerns stay in components/.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

from config.settings import (
    CLASS_NAMES, TARGET_SIZE,
    TTA_FIRST_WEIGHT, TTA_OTHER_WEIGHT,
)
from utils.model_architecture import FusionBackboneClassifier


# ── Face detection ────────────────────────────────────────────────────────────

def detect_and_crop_face(
    image_pil: Image.Image,
    mtcnn: MTCNN,
) -> tuple[Image.Image, bool]:
    """
    Detect the largest face in `image_pil` and return a cropped PIL image.

    Returns
    -------
    cropped : PIL.Image
        Cropped face (or original image when no face is found).
    face_found : bool
        True when at least one face was detected.
    """
    boxes, _ = mtcnn.detect(image_pil)

    if boxes is not None and len(boxes) > 0:
        # Pick the face with the largest bounding-box area
        areas    = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best_box = boxes[int(np.argmax(areas))]

        w, h = image_pil.size
        x1 = max(0, int(best_box[0]))
        y1 = max(0, int(best_box[1]))
        x2 = min(w,  int(best_box[2]))
        y2 = min(h,  int(best_box[3]))

        if x1 < x2 and y1 < y2:
            cropped = image_pil.crop((x1, y1, x2, y2))
            if cropped.size[0] > 0 and cropped.size[1] > 0:
                return cropped, True

    # Fallback — no face detected
    return image_pil, False


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_single(
    image_np: np.ndarray,
    model: FusionBackboneClassifier,
    transform,
) -> np.ndarray:
    """Apply one transform and return softmax probabilities as a 1-D array."""
    tensor = transform(image=image_np)["image"].unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
    return probs[0].cpu().numpy().astype(float)


def predict_single(
    image_np: np.ndarray,
    model: FusionBackboneClassifier,
    transform,
) -> tuple[int, float, np.ndarray]:
    """
    Single-pass prediction (no TTA).

    Returns
    -------
    prediction : int        — index into CLASS_NAMES
    confidence : float      — probability of predicted class
    probs      : np.ndarray — per-class probabilities
    """
    probs      = _run_single(image_np, model, transform)
    prediction = int(np.argmax(probs))
    return prediction, float(probs[prediction]), probs


def predict_with_tta(
    image_np: np.ndarray,
    model: FusionBackboneClassifier,
    tta_transforms: list,
) -> tuple[int, float, np.ndarray]:
    """
    Weighted ensemble over TTA transforms.
    The first transform (original image) is weighted 1.5×, others 1.0×.

    Returns
    -------
    prediction : int        — index into CLASS_NAMES
    confidence : float      — probability of predicted class
    probs      : np.ndarray — per-class probabilities
    """
    weights = [
        TTA_FIRST_WEIGHT if i == 0 else TTA_OTHER_WEIGHT
        for i in range(len(tta_transforms))
    ]

    weighted_sum = sum(
        w * _run_single(image_np, model, tf)
        for w, tf in zip(weights, tta_transforms)
    )
    avg_probs  = weighted_sum / sum(weights)
    prediction = int(np.argmax(avg_probs))
    return prediction, float(avg_probs[prediction]), avg_probs


# ── Result helpers ────────────────────────────────────────────────────────────

def confidence_label(confidence: float) -> tuple[str, str]:
    """
    Return a (label, emoji) pair describing the confidence level.
    Used in the interpretation section of the UI.
    """
    from config.settings import CONF_VERY_HIGH, CONF_HIGH, CONF_MEDIUM
    if confidence > CONF_VERY_HIGH:
        return "sangat tinggi", "🎯"
    if confidence > CONF_HIGH:
        return "tinggi", "✅"
    if confidence > CONF_MEDIUM:
        return "sedang", "⚠️"
    return "rendah", "❓"
