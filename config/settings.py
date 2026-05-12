"""
config/settings.py
All app-wide constants and configuration in one place.
"""

# ── Page ─────────────────────────────────────────────────────────────────────
PAGE_CONFIG = dict(
    page_title="Autism Classification",
    page_icon="🧠",
    layout="wide",
)

# ── Model ────────────────────────────────────────────────────────────────────
BACKBONE_NAME   = "mobilevitv2_100"
FUSION_DIM      = 768
OUT_INDICES     = (1, 2, 3)
NUM_CLASSES     = 2
TARGET_SIZE     = 224
CLASS_NAMES     = ["Autistic", "Non_Autistic"]

FUSION_DROPOUT      = 0.4
CLASSIFIER_DROPOUT  = 0.25

# ── HuggingFace Hub ──────────────────────────────────────────────────────────
HF_REPO_ID   = "Artz-03/autismeClassification"
HF_FILENAME  = "mobilevitv2_phase2_optimized.pth"

# ── Image normalization (ImageNet) ───────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── MTCNN face detector ───────────────────────────────────────────────────────
MTCNN_CONFIG = dict(
    image_size    = TARGET_SIZE,
    margin        = 0,
    min_face_size = 20,
    thresholds    = [0.6, 0.7, 0.7],
    factor        = 0.709,
    post_process  = False,
    device        = "cpu",
)

# ── TTA weights ───────────────────────────────────────────────────────────────
TTA_FIRST_WEIGHT  = 1.5   # weight for the original (no-aug) transform
TTA_OTHER_WEIGHT  = 1.0   # weight for every subsequent transform

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONF_VERY_HIGH = 0.90
CONF_HIGH      = 0.75
CONF_MEDIUM    = 0.60
