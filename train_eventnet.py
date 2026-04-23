"""Train HeatmapEventNet — TTNet-inspired ball event classifier.

Pipeline:
  1. Build heatmap samples from ball + event annotations (smooth labeling ±3 frames)
  2. Merge OpenTTGames extra samples (bounce only)
  3. Train with BCEWithLogitsLoss + per-sample weights
  4. Save best checkpoint to weights/eventnet_best.pt
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eventnet.dataset import IDX_TO_LABEL, NUM_CLASSES
from eventnet.heatmap import HeatmapEventDataset, build_heatmap_samples
from eventnet.model import HeatmapEventNetConfig
from eventnet.openttgames import build_openttgames_heatmap_samples
from eventnet.train import run_training
from webapp.ball_storage import load_ball_annotations
from webapp.storage import load_annotations

DATASET_DIR = Path("dataset")
WEIGHTS_DIR = Path("weights")

BALL_ANNOTATIONS_PATH  = DATASET_DIR / "ball_annotations.json"
EVENT_ANNOTATIONS_PATH = DATASET_DIR / "annotations.json"
OPENTTGAMES_DIR        = DATASET_DIR / "openttgames"

WINDOW_SIZE  = 15
FRAME_WIDTH  = 1920.0
FRAME_HEIGHT = 1080.0
EPOCHS       = 300
LR           = 3e-4
BATCH_SIZE   = 64
VAL_RATIO    = 0.2


def entrypoint() -> None:
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )

    ball_store  = load_ball_annotations(BALL_ANNOTATIONS_PATH)
    event_store = load_annotations(EVENT_ANNOTATIONS_PATH)

    cfg = HeatmapEventNetConfig(window_size=WINDOW_SIZE)

    own_samples = build_heatmap_samples(
        ball_store=ball_store,
        event_store=event_store,
        window_size=WINDOW_SIZE,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )
    print(f"Own samples: {len(own_samples)}")

    extra_samples = []
    if OPENTTGAMES_DIR.exists():
        extra_samples = build_openttgames_heatmap_samples(
            data_dir=OPENTTGAMES_DIR,
            window_size=WINDOW_SIZE,
        )
    else:
        print(
            f"OpenTTGames not found at {OPENTTGAMES_DIR}.\n"
            "  Run: uv run python download_openttgames.py"
        )

    samples = own_samples + extra_samples
    print(f"Total samples: {len(samples)}")

    counts = {
        IDX_TO_LABEL[i]: sum(1 for s in samples if s.label_idx == i)
        for i in range(NUM_CLASSES)
    }
    print(f"Label distribution: {counts}")

    if len(samples) < 4:
        raise RuntimeError("Not enough samples to train. Annotate more events first.")

    n_val   = max(1, int(len(samples) * VAL_RATIO))
    n_train = len(samples) - n_val
    g = torch.Generator().manual_seed(42)
    idx_all    = torch.randperm(len(samples), generator=g).tolist()
    train_samp = [samples[i] for i in idx_all[:n_train]]
    val_samp   = [samples[i] for i in idx_all[n_train:]]

    train_ds = HeatmapEventDataset(train_samp, dropout_p=0.50)
    val_ds   = HeatmapEventDataset(val_samp,   dropout_p=0.0)

    pin = device.type == "cuda"
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin,
    )

    run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        output_dir=WEIGHTS_DIR,
    )


if __name__ == "__main__":
    entrypoint()
