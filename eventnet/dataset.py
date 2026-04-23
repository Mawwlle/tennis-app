"""EventNet dataset: feature extraction from ball trajectory + net geometry."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from model.schemas import AnnotationStore, BallAnnotationStore
from eventnet.features import N_FEATURES, NetGeometry, extract_features
from eventnet.player_detection import PlayerGeometry


LABEL_TO_IDX: dict[str, int] = {
    "bounce": 0,
    "none": 1,
}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES: int = 2


@dataclass(frozen=True)
class EventSample:
    # Normalized (cx, cy) in [0, 1]; None = invisible / not annotated
    positions: tuple[tuple[float, float] | None, ...]
    label_idx: int
    net_geometry: NetGeometry = field(default_factory=NetGeometry)
    player_geometry: PlayerGeometry = field(default_factory=PlayerGeometry)


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
    net_geometries: dict[str, NetGeometry] | None = None,
    player_geometries: dict[str, PlayerGeometry] | None = None,
    neg_ratio: float = 2.0,
    rng_seed: int = 42,
) -> list[EventSample]:
    """Build positive (event) and negative (background) training samples.

    Args:
        net_geometries:    Optional mapping from video_id → NetGeometry.
        player_geometries: Optional mapping from video_id → PlayerGeometry.
                           Videos without an entry use default geometry.
    """
    rng = random.Random(rng_seed)
    half = window_size // 2
    samples: list[EventSample] = []
    geos    = net_geometries    or {}
    players = player_geometries or {}

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

        geo    = geos.get(video_id, NetGeometry())
        player = players.get(video_id, PlayerGeometry())
        event_frames = {a.frame_idx for a in event_annots}

        for annot in event_annots:
            if annot.label not in LABEL_TO_IDX:
                continue
            window = _extract_window(frame_map, annot.frame_idx, half, frame_width, frame_height)
            if window is not None:
                samples.append(EventSample(
                    positions=tuple(window),
                    label_idx=LABEL_TO_IDX[annot.label],
                    net_geometry=geo,
                    player_geometry=player,
                ))

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
            window = _extract_window(frame_map, frame_idx, half, frame_width, frame_height)
            if window is not None:
                samples.append(EventSample(
                    positions=tuple(window),
                    label_idx=LABEL_TO_IDX["none"],
                    net_geometry=geo,
                    player_geometry=player,
                ))

    return samples


class KinematicEventDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, samples: list[EventSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        s = self._samples[idx]
        feats = extract_features(s.positions, s.net_geometry, s.player_geometry)  # (N, 14)
        return torch.from_numpy(feats.T), s.label_idx                              # (14, N), int
