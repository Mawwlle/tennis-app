"""PyTorch dataset for TrackNet training."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tracknet.heatmap import make_empty_heatmap, make_heatmap

TARGET_W   = 640
TARGET_H   = 360
FRAMES_DIR = Path("dataset/frames")


@dataclass
class Sample:
    video_path: Path
    frame_idx: int      # frame t (annotated)
    cx: float           # ball cx in original image coords
    cy: float           # ball cy in original image coords
    orig_w: int
    orig_h: int
    visibility: int     # 1 = visible, 0 = not in frame


def _frame_path(video_path: Path, frame_idx: int) -> Path:
    """dataset/frames/{category}/{videoname_no_ext}/{idx:06d}.jpg"""
    rel = video_path.with_suffix("").name
    category = video_path.parent.name
    return FRAMES_DIR / category / rel / f"{frame_idx:06d}.jpg"


def prepare_frames(samples: list[Sample]) -> None:
    """Pre-extract all needed frames from videos to JPEG files on disk.

    Only extracts frames that are referenced in the sample list
    (t-2, t-1, t for each annotated frame). Skips already-extracted files.
    """
    # Collect needed indices per video
    needed: dict[Path, set[int]] = defaultdict(set)
    for s in samples:
        t = s.frame_idx
        needed[s.video_path].update([max(0, t - 2), max(0, t - 1), t])

    total = sum(len(v) for v in needed.values())
    already = sum(1 for vp, idxs in needed.items() for i in idxs if _frame_path(vp, i).exists())
    if already == total:
        print(f"  Frames already extracted ({total} files)")
        return

    print(f"  Extracting {total - already} frames (skipping {already} existing)…")

    with tqdm(total=total - already, unit="frame") as bar:
        for video_path, idxs in needed.items():
            cap = cv2.VideoCapture(str(video_path))
            for idx in sorted(idxs):
                dst = _frame_path(video_path, idx)
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
                    cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                bar.update()
            cap.release()


def _load_frame(video_path: Path, idx: int) -> NDArray[np.float32]:
    """Read a pre-extracted frame. Falls back to black frame on failure."""
    path = _frame_path(video_path, idx)
    frame = cv2.imread(str(path))
    if frame is None:
        return np.zeros((TARGET_H, TARGET_W, 3), dtype=np.float32)
    return frame.astype(np.float32) / 255.0


class TrackNetDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        s = self.samples[idx]
        t = s.frame_idx
        frames = [_load_frame(s.video_path, max(0, t - 2)),
                  _load_frame(s.video_path, max(0, t - 1)),
                  _load_frame(s.video_path, t)]

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
    # Pre-extract frames once — after this __getitem__ is just imread
    prepare_frames(samples)

    # Split per-video by temporal order; last val_ratio% of each video = val
    by_video: dict[Path, list[Sample]] = defaultdict(list)
    for s in samples:
        by_video[s.video_path].append(s)

    train_samples: list[Sample] = []
    val_samples:   list[Sample] = []
    for vid_samples in by_video.values():
        vid_samples.sort(key=lambda s: s.frame_idx)
        cut = max(1, int(len(vid_samples) * (1 - val_ratio)))
        train_samples.extend(vid_samples[:cut])
        val_samples.extend(vid_samples[cut:])

    # Drop samples whose extracted frame is missing (unreadable video)
    train_samples = [s for s in train_samples if _frame_path(s.video_path, s.frame_idx).exists()]
    val_samples   = [s for s in val_samples   if _frame_path(s.video_path, s.frame_idx).exists()]

    print(f"Dataset: {len(train_samples)} train, {len(val_samples)} val samples")

    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        TrackNetDataset(train_samples), batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        TrackNetDataset(val_samples), batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader
