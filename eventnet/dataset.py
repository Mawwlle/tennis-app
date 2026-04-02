"""EventNet dataset: builds 9-heatmap windows from ball + event annotations."""

import random
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from model.schemas import BallAnnotationStore, AnnotationStore
from tracknet.heatmap import make_heatmap, make_empty_heatmap


LABEL_TO_IDX: dict[str, int] = {
    "hit": 0,
    "bounce": 1,
    "net": 2,
    "none": 3,
}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES: int = 4


@dataclass(frozen=True)
class HeatmapEventSample:
    # Normalized (cx, cy) in [0, 1]; None = ball invisible / not annotated
    positions: tuple[tuple[float, float] | None, ...]
    label_idx: int


def _extract_window(
    frame_map: dict[int, tuple[float, float]],
    center: int,
    half: int,
    frame_width: float,
    frame_height: float,
) -> list[tuple[float, float] | None] | None:
    """Extract 2*half+1 normalized positions centered at `center`.

    Returns None when >50% of the window frames are missing.
    """
    n = 2 * half + 1
    window: list[tuple[float, float] | None] = []
    missing = 0
    for idx in range(center - half, center + half + 1):
        if idx in frame_map:
            cx, cy = frame_map[idx]
            window.append((cx / frame_width, cy / frame_height))
        else:
            window.append(None)
            missing += 1
    if missing > n // 2:
        return None
    return window


def build_samples(
    ball_store: BallAnnotationStore,
    event_store: AnnotationStore,
    window_size: int,
    frame_width: float,
    frame_height: float,
    neg_ratio: float = 2.0,
    rng_seed: int = 42,
) -> list[HeatmapEventSample]:
    """Build positive (event) and negative (background) training samples."""
    rng = random.Random(rng_seed)
    half = window_size // 2
    samples: list[HeatmapEventSample] = []

    for video_id, event_annots in event_store.videos.items():
        ball_annots = ball_store.videos.get(video_id, [])
        if not ball_annots:
            continue

        frame_map: dict[int, tuple[float, float]] = {
            a.frame_idx: (a.cx, a.cy)
            for a in ball_annots
            if a.visibility == 1
        }
        if not frame_map:
            continue

        event_frames = {a.frame_idx for a in event_annots}

        for annot in event_annots:
            window = _extract_window(
                frame_map, annot.frame_idx, half, frame_width, frame_height
            )
            if window is not None:
                samples.append(
                    HeatmapEventSample(
                        positions=tuple(window),
                        label_idx=LABEL_TO_IDX[annot.label],
                    )
                )

        all_frames = sorted(frame_map.keys())
        candidates = [
            f for f in all_frames
            if all(abs(f - ef) > half for ef in event_frames)
            and f >= all_frames[0] + half
            and f <= all_frames[-1] - half
        ]
        n_neg = int(len(event_annots) * neg_ratio)
        chosen = rng.sample(candidates, min(n_neg, len(candidates)))
        for frame_idx in chosen:
            window = _extract_window(
                frame_map, frame_idx, half, frame_width, frame_height
            )
            if window is not None:
                samples.append(
                    HeatmapEventSample(
                        positions=tuple(window),
                        label_idx=LABEL_TO_IDX["none"],
                    )
                )

    return samples


def make_heatmaps(
    positions: tuple[tuple[float, float] | None, ...],
    heatmap_w: int,
    heatmap_h: int,
    sigma: float,
) -> NDArray[np.float32]:
    """Convert normalized positions to stacked heatmaps (N, H, W)."""
    maps: list[NDArray[np.float32]] = []
    for pos in positions:
        if pos is None:
            maps.append(make_empty_heatmap(heatmap_w, heatmap_h))
        else:
            cx_norm, cy_norm = pos
            maps.append(
                make_heatmap(
                    cx_norm * heatmap_w,
                    cy_norm * heatmap_h,
                    heatmap_w,
                    heatmap_h,
                    sigma=sigma,
                )
            )
    return np.stack(maps)  # (N, H, W)


class HeatmapEventDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        samples: list[HeatmapEventSample],
        heatmap_w: int,
        heatmap_h: int,
        sigma: float,
    ) -> None:
        self._samples = samples
        self._heatmap_w = heatmap_w
        self._heatmap_h = heatmap_h
        self._sigma = sigma

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        s = self._samples[idx]
        heatmaps = make_heatmaps(s.positions, self._heatmap_w, self._heatmap_h, self._sigma)
        return torch.from_numpy(heatmaps), s.label_idx  # (N, H, W), int
