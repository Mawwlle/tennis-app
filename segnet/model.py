"""SegNet: encoder-decoder segmentation network.

Encoder: 5 VGG-style blocks with max-pooling (indices saved).
Decoder: 5 symmetric blocks with max-unpooling + conv.
Output:  per-pixel logits (B, num_classes, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _enc_block(in_ch: int, out_ch: int, n_conv: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n_conv):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


def _dec_block(in_ch: int, out_ch: int, n_conv: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n_conv):
        ch_in = in_ch if i == 0 else out_ch
        layers += [
            nn.Conv2d(ch_in, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class SegNet(nn.Module):
    """SegNet with 5 encoder/decoder stages (VGG16 channel layout)."""

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()

        # Encoder blocks
        self.enc1 = _enc_block(in_channels, 64, 2)
        self.enc2 = _enc_block(64, 128, 2)
        self.enc3 = _enc_block(128, 256, 3)
        self.enc4 = _enc_block(256, 512, 3)
        self.enc5 = _enc_block(512, 512, 3)

        # Decoder blocks (mirror)
        self.dec5 = _dec_block(512, 512, 3)
        self.dec4 = _dec_block(512, 256, 3)
        self.dec3 = _dec_block(256, 128, 3)
        self.dec2 = _dec_block(128, 64, 2)
        self.dec1 = _dec_block(64, 64, 2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — save input sizes before each pool for exact unpool reconstruction
        e1 = self.enc1(x);  s1 = e1.size(); x, idx1 = self.pool(e1)
        e2 = self.enc2(x);  s2 = e2.size(); x, idx2 = self.pool(e2)
        e3 = self.enc3(x);  s3 = e3.size(); x, idx3 = self.pool(e3)
        e4 = self.enc4(x);  s4 = e4.size(); x, idx4 = self.pool(e4)
        e5 = self.enc5(x);  s5 = e5.size(); x, idx5 = self.pool(e5)

        # Decoder — pass output_size to handle odd spatial dimensions
        x = self.dec5(self.unpool(x, idx5, output_size=s5))
        x = self.dec4(self.unpool(x, idx4, output_size=s4))
        x = self.dec3(self.unpool(x, idx3, output_size=s3))
        x = self.dec2(self.unpool(x, idx2, output_size=s2))
        x = self.dec1(self.unpool(x, idx1, output_size=s1))

        return self.classifier(x)  # type: ignore[no-any-return]
