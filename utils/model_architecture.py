"""
utils/model_architecture.py
Neural network definitions — mirrors training code exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SELayer(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OptimizedFeatureFusionBlock(nn.Module):
    """
    Fuses multi-scale feature maps from different backbone stages.
    Each feature map is projected to `output_channels`, then combined
    via learned gates + SE refinement. Returns a 1-D pooled vector.
    """

    def __init__(
        self,
        input_channels: list[int],
        output_channels: int = 512,
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        # Per-stage projection: C_i → output_channels
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.SiLU(inplace=True),
            )
            for c in input_channels
        ])

        # Gating: select which projected features to amplify
        self.gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_channels * len(input_channels), output_channels, 1),
            nn.Sigmoid(),
        )

        self.se_block   = SELayer(output_channels, reduction=8)
        self.fuse_conv  = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.fuse_bn    = nn.BatchNorm2d(output_channels)
        self.dropout    = nn.Dropout(dropout_rate)

    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = feature_maps[-1].shape[2:]

        # Upsample / downsample each stage to the same spatial size, then project
        projected = [
            proj(F.interpolate(fmap, size=(target_h, target_w), mode="bilinear", align_corners=False))
            for fmap, proj in zip(feature_maps, self.projs)
        ]

        # Gated fusion
        gate  = self.gate_gen(torch.cat(projected, dim=1))
        fused = sum(projected) * gate

        # SE refinement with residual
        refined = self.fuse_bn(self.fuse_conv(fused))
        fused   = F.silu(refined + fused)
        fused   = self.se_block(fused)

        # Global pooling: GAP + GMP → sum
        gap = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(fused, 1).flatten(1)
        return self.dropout(gap + gmp)


class FusionBackboneClassifier(nn.Module):
    """
    Main classifier.
      backbone → multi-scale features → OptimizedFeatureFusionBlock → Linear head
    """

    def __init__(
        self,
        backbone_name: str  = "mobilevitv2_100",
        out_indices: tuple  = (1, 2, 3),
        fusion_dim: int     = 768,
        num_classes: int    = 2,
        fusion_dropout: float     = 0.4,
        classifier_dropout: float = 0.25,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=out_indices,
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
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats     = self.backbone(x)
        fused_vec = self.fusion(feats)
        return self.classifier(fused_vec)
