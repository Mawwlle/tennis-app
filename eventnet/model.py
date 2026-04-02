"""HeatmapEventNet: 2D CNN classifier over a window of 9 ball heatmaps."""

import torch
from torch import Tensor, nn
from pydantic import BaseModel


class HeatmapEventNetConfig(BaseModel):
    window_size: int = 9       # number of consecutive heatmaps
    heatmap_w: int = 64
    heatmap_h: int = 36
    num_classes: int = 4       # hit, bounce, net, none
    frame_width: float = 1920.0   # coordinate space of training annotations
    frame_height: float = 1080.0
    sigma: float = 3.0         # Gaussian sigma in heatmap pixels


class HeatmapEventNet(nn.Module):
    """Classify game events from 9 consecutive ball heatmaps.

    Input:  (B, 9, H, W)     — stacked Gaussian heatmaps, one per frame
    Output: (B, num_classes) — logits for [hit, bounce, net, none]

    Treats the 9 heatmaps like 9 channels for 2D convolution.
    Each heatmap encodes the ball position as a Gaussian blob, so
    the conv layers learn spatial trajectory patterns.
    """

    def __init__(self, cfg: HeatmapEventNetConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.window_size, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # H/2, W/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # H/4, W/4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(32, cfg.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))   # (B, 9, H, W) → (B, num_classes)
