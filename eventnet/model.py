"""EventNet models.

HeatmapEventNet (primary) — TTNet-inspired:
  Input:  (B, window_size, HM_H, HM_W) — stacked ball detection heatmaps
  Per-frame spatial encoder (shared CNN) → 32-dim feature vector per frame
  Temporal TCN (3 dilated blocks) → (B, num_classes) logits
  Trained with BCEWithLogitsLoss + smooth target labeling

TCNEventNet (legacy) — kinematic scalar features.
"""

import torch.nn.functional as F
from torch import Tensor, nn
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------


class _TCNBlock(nn.Module):
    """Dilated conv1d + residual skip."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.conv(x) + self.skip(x))


# ---------------------------------------------------------------------------
# HeatmapEventNet (primary — TTNet approach)
# ---------------------------------------------------------------------------


class HeatmapEventNetConfig(BaseModel):
    window_size: int = 15
    heatmap_h: int = 36
    heatmap_w: int = 64
    spatial_channels: int = 32   # output dim of per-frame spatial encoder
    tcn_channels: int = 64
    num_classes: int = 2          # bounce, none


class HeatmapEventNet(nn.Module):
    """Event classifier from stacked ball-detection heatmaps.

    Architecture (inspired by TTNet's event spotting branch):
      Shared spatial encoder: small 2D CNN → global pool → spatial_channels features
      Temporal aggregation: 3-block dilated TCN over the window dimension
      Head: pool → dropout → linear → logits

    Input:  (B, window_size, H, W)  e.g. (B, 15, 36, 64)
    Output: (B, num_classes)        raw logits for BCEWithLogitsLoss

    Params: ~100 K   Inference: < 3 ms on CPU per window.
    """

    def __init__(self, cfg: HeatmapEventNetConfig) -> None:
        super().__init__()
        ch = cfg.spatial_channels

        # Per-frame encoder: 36×64 → ch-dim vector
        # Two stride-2 layers → 9×16 → global avg pool
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),  # 18×32
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, ch, kernel_size=3, stride=2, padding=1),  # 9×16
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                                            # (ch,)
        )

        # Temporal TCN: (B, ch, window_size) → (B, tcn_channels, window_size)
        tc = cfg.tcn_channels
        self.tcn = nn.Sequential(
            _TCNBlock(ch, tc, dilation=1),
            _TCNBlock(tc, tc, dilation=2),
            _TCNBlock(tc, tc, dilation=4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(tc, cfg.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, W, H, W_hm)
        B, W, H_hm, W_hm = x.shape
        x_flat = x.view(B * W, 1, H_hm, W_hm)
        feats  = self.spatial(x_flat)           # (B*W, ch)
        feats  = feats.view(B, W, -1)           # (B, W, ch)
        feats  = feats.permute(0, 2, 1)         # (B, ch, W)
        return self.head(self.tcn(feats))        # (B, num_classes)


# ---------------------------------------------------------------------------
# TCNEventNet (legacy — kinematic features)
# ---------------------------------------------------------------------------


class TCNEventNetConfig(BaseModel):
    window_size: int = 15
    n_features: int = 14
    channels: int = 48
    num_classes: int = 3       # hit, bounce, none
    frame_width: float = 1920.0
    frame_height: float = 1080.0


class TCNEventNet(nn.Module):
    def __init__(self, cfg: TCNEventNetConfig) -> None:
        super().__init__()
        ch = cfg.channels
        self.tcn = nn.Sequential(
            _TCNBlock(cfg.n_features, ch, dilation=1),
            _TCNBlock(ch,             ch, dilation=2),
            _TCNBlock(ch,             ch, dilation=4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(ch, cfg.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.tcn(x))
