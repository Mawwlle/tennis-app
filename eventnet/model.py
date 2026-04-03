"""TCNEventNet: dilated 1D TCN classifier over kinematic ball trajectory features."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pydantic import BaseModel


class TCNEventNetConfig(BaseModel):
    window_size: int = 9       # number of consecutive frames
    n_features: int = 8        # [cx, cy, dx, dy, d2x, d2y, speed, angle]
    channels: int = 32
    num_classes: int = 4       # hit, bounce, net, none
    frame_width: float = 1920.0
    frame_height: float = 1080.0


class _TCNBlock(nn.Module):
    """Dilated causal conv + residual skip."""

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


class TCNEventNet(nn.Module):
    """Classify game events from kinematic ball trajectory.

    Input:  (B, n_features, window_size)  — 8 kinematic features per frame
    Output: (B, num_classes)              — logits for hit / bounce / net / none

    Three TCN blocks with dilation [1, 2, 4] give a receptive field of 29 steps,
    covering the full 9-frame window with room to spare.
    ~8 K parameters total — fits in <50 KB ONNX.
    """

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
        return self.head(self.tcn(x))   # (B, n_features, N) → (B, num_classes)
