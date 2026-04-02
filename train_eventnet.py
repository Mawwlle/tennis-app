"""Entry point for HeatmapEventNet game event classifier training."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from eventnet.dataset import (
    HeatmapEventDataset,
    IDX_TO_LABEL,
    NUM_CLASSES,
    build_samples,
)
from eventnet.model import HeatmapEventNetConfig
from eventnet.train import run_training
from webapp.ball_storage import load_ball_annotations
from webapp.storage import load_annotations

DATASET_DIR = Path("dataset")
WEIGHTS_DIR = Path("weights")

BALL_ANNOTATIONS_PATH = DATASET_DIR / "ball_annotations.json"
EVENT_ANNOTATIONS_PATH = DATASET_DIR / "annotations.json"

WINDOW_SIZE  = 9
FRAME_WIDTH  = 1920.0
FRAME_HEIGHT = 1080.0
HEATMAP_W    = 64
HEATMAP_H    = 36
SIGMA        = 3.0
EPOCHS       = 300
LR           = 1e-3
BATCH_SIZE   = 32
VAL_RATIO    = 0.2


def entrypoint() -> None:
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )

    ball_store  = load_ball_annotations(BALL_ANNOTATIONS_PATH)
    event_store = load_annotations(EVENT_ANNOTATIONS_PATH)

    cfg = HeatmapEventNetConfig(
        window_size=WINDOW_SIZE,
        heatmap_w=HEATMAP_W,
        heatmap_h=HEATMAP_H,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        sigma=SIGMA,
    )

    samples = build_samples(
        ball_store=ball_store,
        event_store=event_store,
        window_size=WINDOW_SIZE,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )

    print(f"Total samples: {len(samples)}")
    counts = {IDX_TO_LABEL[i]: sum(1 for s in samples if s.label_idx == i) for i in range(NUM_CLASSES)}
    print(f"Label distribution: {counts}")

    if len(samples) < 4:
        raise RuntimeError("Not enough samples to train. Annotate more events first.")

    # Inverse-frequency class weights to counter class imbalance
    total = len(samples)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * max(counts[IDX_TO_LABEL[i]], 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    )
    print(f"Class weights: { {IDX_TO_LABEL[i]: f'{class_weights[i]:.2f}' for i in range(NUM_CLASSES)} }")

    dataset = HeatmapEventDataset(samples, HEATMAP_W, HEATMAP_H, SIGMA)
    n_val   = max(1, int(len(dataset) * VAL_RATIO))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
    )

    run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        class_weights=class_weights,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        output_dir=WEIGHTS_DIR,
    )


if __name__ == "__main__":
    entrypoint()
