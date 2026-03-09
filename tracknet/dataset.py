"""PyTorch dataset for TrackNet training."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tracknet.heatmap import make_empty_heatmap, make_heatmap

TARGET_W = 640
TARGET_H = 360


@dataclass
class Sample:
    video_path: Path
    frame_idx: int      # frame t (annotated)
    cx: float           # ball cx in original image coords
    cy: float           # ball cy in original image coords
    orig_w: int
    orig_h: int
    visibility: int     # 1 = visible, 0 = not in frame


def _read_frame(cap: cv2.VideoCapture, idx: int) -> NDArray[np.float32] | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
    return frame.astype(np.float32) / 255.0


def is_sample_readable(s: Sample) -> bool:
    """Return True only if the annotated frame (t) is actually readable."""
    cap = cv2.VideoCapture(str(s.video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, s.frame_idx)
    ret, _ = cap.read()
    cap.release()
    return ret


class TrackNetDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        s = self.samples[idx]
        cap = cv2.VideoCapture(str(s.video_path))

        # Triplet: t-2, t-1, t (clamp to valid range)
        t = s.frame_idx
        idxs = [max(0, t - 2), max(0, t - 1), t]
        raw_frames = [_read_frame(cap, i) for i in idxs]
        cap.release()

        # Replace any unreadable frames with the nearest readable one
        fallback = next((f for f in raw_frames if f is not None), None)
        if fallback is None:
            fallback = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.float32)
        frames = [f if f is not None else fallback for f in raw_frames]

        # Stack: (3, H, W, 3) → (9, H, W)  via  (H, W, 3)*3 → concat on channel
        stacked = np.concatenate(
            [f.transpose(2, 0, 1) for f in frames], axis=0
        ).astype(np.float32)

        if s.visibility == 1:
            cx_s = s.cx * TARGET_W / s.orig_w
            cy_s = s.cy * TARGET_H / s.orig_h
            heatmap = make_heatmap(cx_s, cy_s, TARGET_W, TARGET_H)
        else:
            heatmap = make_empty_heatmap(TARGET_W, TARGET_H)

        return torch.from_numpy(stacked), torch.from_numpy(heatmap).unsqueeze(0)


def build_loaders(
    samples: list[Sample],
    val_ratio: float,
    batch_size: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    # Filter out samples whose annotated frame can't be read from the video
    valid = [s for s in samples if is_sample_readable(s)]
    dropped = len(samples) - len(valid)
    if dropped:
        print(f"  WARNING: dropped {dropped} unreadable samples")
    samples = valid

    # Split per-video to keep temporal order; use last val_ratio of each video
    from collections import defaultdict
    by_video: dict[Path, list[Sample]] = defaultdict(list)
    for s in samples:
        by_video[s.video_path].append(s)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    for vid_samples in by_video.values():
        vid_samples.sort(key=lambda s: s.frame_idx)
        cut = max(1, int(len(vid_samples) * (1 - val_ratio)))
        train_samples.extend(vid_samples[:cut])
        val_samples.extend(vid_samples[cut:])

    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        TrackNetDataset(train_samples), batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        TrackNetDataset(val_samples), batch_size=batch_size, shuffle=False, num_workers=2
    )
    print(f"Dataset: {len(train_samples)} train, {len(val_samples)} val samples")
    return train_loader, val_loader