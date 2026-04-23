"""Heatmap-based dataset for EventNet, following TTNet's approach.

Key ideas from TTNet (Voeikov et al., CVPRW 2020):
  - Use ball detection heatmaps as input (richer spatial context than scalar cx/cy)
  - Sin-based smooth target labeling over ±SMOOTH_RADIUS frames around each event
  - BCEWithLogitsLoss — each class is independent sigmoid (not softmax)
  - Training: synthetic Gaussian blobs from ball annotations
  - Inference: actual TrackNet output heatmaps (already computed in Pass 1)

Smooth labeling: for event at frame F with radius R, windows centered at F+d
(d ∈ {-R…R}) get target weight = sin((R - |d| + 1) * π / (2*(R+1))).
  d=0 → 1.0,  d=±1 → 0.92,  d=±2 → 0.71,  d=±3 → 0.38  (for R=3)

This gives ~7× more hit training samples and makes the model robust to
annotation timing imprecision (±1–2 frames).

Heatmap dropout: during training, randomly zero 50% of frames per window to
simulate sparse TrackNet inference (INFER_STEP=3 → only every 3rd frame).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from eventnet.dataset import LABEL_TO_IDX, NUM_CLASSES, _extract_window
from model.schemas import AnnotationStore, BallAnnotationStore

# Heatmap spatial resolution (10× downscale from 360×640)
HM_H: int = 36
HM_W: int = 64
HM_SIGMA: float = 2.0   # Gaussian radius in heatmap pixels

# Smooth label parameters
SMOOTH_RADIUS: int = 3   # ± frames around each event to create samples


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------


def make_gaussian_heatmap(
    cx_norm: float | None,
    cy_norm: float | None,
    h: int = HM_H,
    w: int = HM_W,
    sigma: float = HM_SIGMA,
) -> NDArray[np.float32]:
    """Gaussian blob at normalized (cx, cy). Returns zeros if position is None."""
    hm = np.zeros((h, w), dtype=np.float32)
    if cx_norm is None or cy_norm is None:
        return hm
    px = cx_norm * w
    py = cy_norm * h
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None]
    hm = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2.0 * sigma ** 2))
    return hm.astype(np.float32)


def positions_to_heatmaps(
    positions: tuple[tuple[float, float] | None, ...],
    h: int = HM_H,
    w: int = HM_W,
    sigma: float = HM_SIGMA,
) -> NDArray[np.float32]:
    """Convert a window of normalized (cx, cy) → stacked heatmaps (W, H, W_hm)."""
    return np.stack(
        [
            make_gaussian_heatmap(pos[0] if pos is not None else None,
                                  pos[1] if pos is not None else None, h, w, sigma)
            for pos in positions
        ],
        axis=0,
    )  # (window_size, h, w)


# ---------------------------------------------------------------------------
# Sin-based smooth target weights (TTNet approach)
# ---------------------------------------------------------------------------


def _smooth_weight(distance: int, radius: int = SMOOTH_RADIUS) -> float:
    """sin-based weight for a window offset by `distance` frames from the event.

    Matches TTNet's sin(n*π/8) labeling adapted for our window radius.
    """
    if abs(distance) > radius:
        return 0.0
    return math.sin((radius - abs(distance) + 1) * math.pi / (2 * (radius + 1)))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeatmapEventSample:
    heatmaps: NDArray[np.float32]   # (window_size, HM_H, HM_W)
    label_idx: int
    weight: float = 1.0             # sample weight for weighted BCELoss


class HeatmapEventDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Dataset that returns (heatmaps, one_hot_target, sample_weight).

    Uses BCEWithLogitsLoss: target is a one-hot float vector, weighted by
    smooth_weight. Allows multi-label soft targets matching TTNet's design.
    """

    def __init__(self, samples: list[HeatmapEventSample], dropout_p: float = 0.50) -> None:
        self._samples  = samples
        self._dropout_p = dropout_p   # fraction of frames to zero out per window

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        s = self._samples[idx]
        hm = s.heatmaps.copy()

        # Heatmap dropout: simulate INFER_STEP=3 sparsity seen at inference
        if self._dropout_p > 0:
            drop = np.random.random(len(hm)) < self._dropout_p
            hm[drop] = 0.0

        target = np.zeros(NUM_CLASSES, dtype=np.float32)
        target[s.label_idx] = float(s.weight)

        return (
            torch.from_numpy(hm),                         # (W, H, W_hm)
            torch.from_numpy(target),                     # (NUM_CLASSES,)
            torch.tensor(s.weight, dtype=torch.float32),  # scalar
        )


# ---------------------------------------------------------------------------
# Sample builder (own annotations)
# ---------------------------------------------------------------------------


def build_heatmap_samples(
    ball_store: BallAnnotationStore,
    event_store: AnnotationStore,
    window_size: int,
    frame_width: float,
    frame_height: float,
    smooth_radius: int = SMOOTH_RADIUS,
    neg_ratio: float = 3.0,
    rng_seed: int = 42,
) -> list[HeatmapEventSample]:
    """Build heatmap training samples with TTNet-style smooth temporal labeling.

    For each event at frame F, creates windows at F-smooth_radius … F+smooth_radius
    with sin-based weights. None samples are drawn far from all event frames.
    """
    rng  = random.Random(rng_seed)
    half = window_size // 2
    samples: list[HeatmapEventSample] = []

    for video_id, event_annots in event_store.videos.items():
        ball_annots = ball_store.videos.get(video_id, [])
        if not ball_annots:
            continue

        # Raw pixel coords → normalized in _extract_window
        frame_map: dict[int, tuple[float, float]] = {
            a.frame_idx: (a.cx, a.cy)
            for a in ball_annots
            if a.visibility == 1
        }
        if not frame_map:
            continue

        event_frames = {a.frame_idx for a in event_annots}

        # ── Positive samples with smooth temporal labeling ────────────────────
        for annot in event_annots:
            if annot.label not in LABEL_TO_IDX:
                continue
            label_idx = LABEL_TO_IDX[annot.label]

            for d in range(-smooth_radius, smooth_radius + 1):
                center = annot.frame_idx + d
                window = _extract_window(frame_map, center, half, frame_width, frame_height)
                if window is None:
                    continue
                hm      = positions_to_heatmaps(tuple(window))
                w_smooth = _smooth_weight(d, smooth_radius)
                samples.append(HeatmapEventSample(
                    heatmaps=hm,
                    label_idx=label_idx,
                    weight=w_smooth,
                ))

        # ── Negative (none) samples ───────────────────────────────────────────
        all_frames = sorted(frame_map.keys())
        candidates = [
            f for f in all_frames
            if all(abs(f - ef) > half + smooth_radius for ef in event_frames)
            and f >= all_frames[0] + half
            and f <= all_frames[-1] - half
        ]
        n_neg = int(len(event_annots) * neg_ratio)
        chosen = rng.sample(candidates, min(n_neg, len(candidates)))
        for fi in chosen:
            window = _extract_window(frame_map, fi, half, frame_width, frame_height)
            if window is None:
                continue
            hm = positions_to_heatmaps(tuple(window))
            samples.append(HeatmapEventSample(
                heatmaps=hm,
                label_idx=LABEL_TO_IDX["none"],
                weight=1.0,
            ))

    return samples
