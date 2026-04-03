"""EventNet dataset: kinematic feature extraction from ball trajectory."""

import random
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from model.schemas import AnnotationStore, BallAnnotationStore


LABEL_TO_IDX: dict[str, int] = {
    "hit": 0,
    "bounce": 1,
    "net": 2,
    "none": 3,
}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES: int = 4
N_FEATURES: int = 8   # cx, cy, dx, dy, d2x, d2y, speed, angle


@dataclass(frozen=True)
class EventSample:
    # Normalized (cx, cy) in [0, 1]; None = invisible / not annotated
    positions: tuple[tuple[float, float] | None, ...]
    label_idx: int


def extract_kinematics(
    positions: tuple[tuple[float, float] | None, ...],
) -> NDArray[np.float32]:
    """Convert a window of normalized (cx, cy) positions to kinematic features.

    Returns (N, 8) array: [cx, cy, dx, dy, d²x, d²y, speed, angle].

    Missing frames (None) contribute zero velocity / acceleration at that step.
    Angle is normalised to [-1, 1] via division by π.
    """
    n = len(positions)
    feats = np.zeros((n, N_FEATURES), dtype=np.float32)

    for i, pos in enumerate(positions):
        if pos is not None:
            feats[i, 0], feats[i, 1] = pos

    for i in range(1, n):
        if positions[i] is not None and positions[i - 1] is not None:
            feats[i, 2] = feats[i, 0] - feats[i - 1, 0]   # dx
            feats[i, 3] = feats[i, 1] - feats[i - 1, 1]   # dy

    for i in range(2, n):
        if positions[i] is not None and positions[i - 1] is not None and positions[i - 2] is not None:
            feats[i, 4] = feats[i, 2] - feats[i - 1, 2]   # d²x
            feats[i, 5] = feats[i, 3] - feats[i - 1, 3]   # d²y

    dx, dy = feats[:, 2], feats[:, 3]
    feats[:, 6] = np.sqrt(dx ** 2 + dy ** 2)               # speed
    feats[:, 7] = np.arctan2(dy, dx) / np.pi               # angle ∈ [-1, 1]

    return feats  # (N, 8)


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
) -> list[EventSample]:
    """Build positive (event) and negative (background) training samples."""
    rng = random.Random(rng_seed)
    half = window_size // 2
    samples: list[EventSample] = []

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
            window = _extract_window(frame_map, annot.frame_idx, half, frame_width, frame_height)
            if window is not None:
                samples.append(EventSample(positions=tuple(window), label_idx=LABEL_TO_IDX[annot.label]))

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
                samples.append(EventSample(positions=tuple(window), label_idx=LABEL_TO_IDX["none"]))

    return samples


class KinematicEventDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, samples: list[EventSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        s = self._samples[idx]
        feats = extract_kinematics(s.positions)          # (N, 8)
        return torch.from_numpy(feats.T), s.label_idx    # (8, N) for Conv1d, int
