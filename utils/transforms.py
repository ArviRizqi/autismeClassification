"""
utils/transforms.py
Albumentations pipelines for inference and Test-Time Augmentation (TTA).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.settings import IMAGENET_MEAN, IMAGENET_STD, TARGET_SIZE


def _base_pipeline(*extra_transforms) -> A.Compose:
    """Shared resize → crop → [extra] → normalize → tensor."""
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(TARGET_SIZE, TARGET_SIZE),
        *extra_transforms,
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_standard_transform() -> A.Compose:
    """Single deterministic transform used when TTA is disabled."""
    return _base_pipeline()


def get_tta_transforms() -> list[A.Compose]:
    """
    Four augmented views used for Test-Time Augmentation.
    The first view (original) receives a higher weight during ensembling.
    """
    return [
        _base_pipeline(),                                   # 0 — original (weight 1.5)
        _base_pipeline(A.HorizontalFlip(p=1.0)),            # 1 — mirror
        A.Compose([                                         # 2 — slight zoom-out
            A.Resize(300, 300),
            A.CenterCrop(TARGET_SIZE, TARGET_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        _base_pipeline(A.Rotate(limit=10, p=1.0)),          # 3 — small rotation
    ]
