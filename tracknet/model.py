"""TrackNet: lightweight UNet for ball detection from 3-frame sequences."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _dw_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Depthwise separable conv: spatial (per-channel) + pointwise (channel mix).

    Обычная Conv2d делает две вещи сразу: ищет паттерны В пространстве
    и смешивает каналы. Здесь разбиваем на два шага:
      1. depthwise — 3×3 свёртка отдельно по каждому каналу (groups=in_ch)
      2. pointwise — 1×1 свёртка для смешивания каналов
    Это даёт ~8× меньше параметров при схожем качестве.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),  # spatial
        nn.Conv2d(in_ch, out_ch, 1, bias=False),                           # channel mix
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _conv_block(in_ch: int, out_ch: int, n: int) -> nn.Sequential:
    """n последовательных depthwise separable блоков."""
    return nn.Sequential(*[
        m
        for i in range(n)
        for m in _dw_block(in_ch if i == 0 else out_ch, out_ch)
    ])


class TrackNet(nn.Module):
    """UNet с depthwise separable свёртками и skip connections.

    Вход:  (B, 9, H, W)  — 3 RGB-кадра [t-2, t-1, t], склеенных по каналам.
    Выход: (B, 1, H, W)  — logits тепловой карты (sigmoid → вероятность мяча).

    Архитектура:
      Энкодер сжимает пространство (3× pool 2×) и учится отвечать на вопрос
      "что происходит в этом регионе?". К концу энкодера карта 8× меньше,
      но каждый пиксель кодирует большой контекст — движущееся пятно = мяч.

      Декодер восстанавливает пространственную точность обратно до оригинального
      разрешения. Skip connections передают детали с каждого уровня энкодера,
      чтобы декодер мог точно локализовать мяч в пикселях.

    Каналы: 9 → 16 → 32 → 64 → 128 → 64 → 32 → 16 → 1
    Параметры: 63K
    """

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Энкодер: каждый уровень удваивает каналы, pool уменьшает H/W в 2×
        self.enc1 = _conv_block(9,  16, 2)   # (B,  16, H,   W)
        self.enc2 = _conv_block(16, 32, 2)   # (B,  32, H/2, W/2)
        self.enc3 = _conv_block(32, 64, 2)   # (B,  64, H/4, W/4)
        self.bottleneck = _conv_block(64, 128, 2)  # (B, 128, H/8, W/8)

        # Декодер: upsample 2× + cat(skip) + conv
        self.dec3 = _conv_block(128 + 64, 64, 2)  # cat с enc3
        self.dec2 = _conv_block(64  + 32, 32, 2)  # cat с enc2
        self.dec1 = _conv_block(32  + 16, 16, 2)  # cat с enc1

        # Финальный 1×1 — проецирует 16 каналов в 1 (logit вероятности мяча)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        # Энкодер — сохраняем skip для каждого уровня
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x  = self.bottleneck(self.pool(e3))

        # Декодер — upsample → cat(skip) → conv
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, e3], dim=1))  # знаем ЧТО + детали enc3
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, e2], dim=1))  # уточняем + детали enc2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, e1], dim=1))  # финальная точность

        return self.head(x)  # (B, 1, H, W) logits