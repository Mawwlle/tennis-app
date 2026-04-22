"""Train SegNet on the net/table segmentation dataset.

Usage:
    uv run train-seg
"""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel

from segnet.dataset import build_dataset
from segnet.train import run_training

ANNOTATIONS_DIR = Path("annotations_output")
FRAMES_DIR = ANNOTATIONS_DIR / "frames"
LABELS_DIR = ANNOTATIONS_DIR / "labels"
WEIGHTS_DIR = Path("weights")

# 0=background  1=net  2=table
NUM_CLASSES = 3
IMAGE_SIZE = (360, 640)  # (H, W) — half of 1920x1080

EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-4
VAL_RATIO = 0.2


class SegTrainConfig(BaseModel):
    num_classes: int
    image_size: tuple[int, int]
    epochs: int
    batch_size: int
    lr: float
    val_ratio: float
    weights_dir: Path
    device: str


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def entrypoint() -> None:
    config = SegTrainConfig(
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        val_ratio=VAL_RATIO,
        weights_dir=WEIGHTS_DIR,
        device=detect_device(),
    )

    dataset = build_dataset(FRAMES_DIR, LABELS_DIR, config.image_size)
    print(f"Loaded {len(dataset)} labeled frames")
    print(f"Device: {config.device}")

    run_training(
        dataset=dataset,
        num_classes=config.num_classes,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        val_ratio=config.val_ratio,
        device=torch.device(config.device),
        weights_dir=config.weights_dir,
    )


if __name__ == "__main__":
    entrypoint()
