from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model.schemas import AnnotationStore, GameState, ModelConfig


def _read_frame(cap: cv2.VideoCapture, frame_idx: int, height: int, width: int) -> NDArray[np.float32]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret: bool
    frame: NDArray[np.uint8]
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    frame = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.astype(np.float32) / 255.0


LABEL_TO_IDX: dict[GameState, int] = {
    GameState.net: 0,
    GameState.bounce: 1,
    GameState.hit: 2,
}


class FrameSequenceDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        store: AnnotationStore,
        dataset_dir: Path,
        config: ModelConfig,
    ) -> None:
        self.config = config
        self.samples: list[tuple[Path, int, int]] = []
        half = config.num_frames // 2

        for video_path_str, annotations in store.videos.items():
            video_path = dataset_dir / video_path_str
            if not video_path.exists():
                continue
            for ann in annotations:
                start = ann.frame_idx - half
                label_idx = LABEL_TO_IDX[ann.label]
                self.samples.append((video_path, start, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        video_path, start_frame, label_idx = self.samples[idx]
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: list[NDArray[np.float32]] = []
        for i in range(self.config.num_frames):
            frame_idx = max(0, min(start_frame + i, total_frames - 1))
            frame = _read_frame(cap, frame_idx, self.config.frame_height, self.config.frame_width)
            frames.append(frame)

        cap.release()

        # (num_frames, H, W, 3) -> (num_frames, 3, H, W)
        stacked = np.stack(frames)
        stacked = np.transpose(stacked, (0, 3, 1, 2))
        return torch.from_numpy(stacked), label_idx


def build_dataloader(
    store: AnnotationStore,
    dataset_dir: Path,
    config: ModelConfig,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[Tensor, int]]:
    dataset = FrameSequenceDataset(store, dataset_dir, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
