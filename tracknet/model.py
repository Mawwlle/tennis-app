"""TrackNet: lightweight UNet for ball detection from 3-frame sequences."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int, n: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class TrackNet(nn.Module):
    """UNet encoder–decoder with skip connections.
    Input: (B, 9, H, W). Output: (B, 1, H, W) logits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = _conv_block(9,   32,  2)
        self.enc2 = _conv_block(32,  64,  2)
        self.enc3 = _conv_block(64,  128, 2)
        self.bottleneck = _conv_block(128, 256, 2)

        # Decoder (skip connections double the input channels)
        self.dec3 = _conv_block(256 + 128, 128, 2)
        self.dec2 = _conv_block(128 + 64,   64, 2)
        self.dec1 = _conv_block(64  + 32,   32, 2)

        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x  = self.bottleneck(self.pool(e3))

        x = F.interpolate(x,  scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, e3], dim=1))
        x = F.interpolate(x,  scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x,  scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, e1], dim=1))

        return self.head(x)
