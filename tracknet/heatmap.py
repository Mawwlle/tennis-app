"""Gaussian heatmap generation for ball position."""

import numpy as np
from numpy.typing import NDArray


def make_heatmap(
    cx: float,
    cy: float,
    width: int,
    height: int,
    sigma: float = 5.0,
) -> NDArray[np.float32]:
    """Return a (height, width) float32 heatmap in [0, 1] with a Gaussian at (cx, cy)."""
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    heatmap: NDArray[np.float32] = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))
    return heatmap


def make_empty_heatmap(width: int, height: int) -> NDArray[np.float32]:
    return np.zeros((height, width), dtype=np.float32)
