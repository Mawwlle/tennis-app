"""Entry point for TCNEventNet game event classifier training.

If dataset/openttgames/ exists and contains annotation files, those samples
are merged with the manually-annotated dataset before training.  OpenTTGames
provides extra bounce / net examples to combat the severe class imbalance.

To download OpenTTGames annotations first::

    uv run python download_openttgames.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from eventnet.dataset import (
    IDX_TO_LABEL,
    KinematicEventDataset,
    NUM_CLASSES,
    build_samples,
)
from eventnet.model import TCNEventNetConfig
from eventnet.openttgames import build_openttgames_samples
from eventnet.train import run_training
from webapp.ball_storage import load_ball_annotations
from webapp.storage import load_annotations

DATASET_DIR = Path("dataset")
WEIGHTS_DIR = Path("weights")

BALL_ANNOTATIONS_PATH  = DATASET_DIR / "ball_annotations.json"
EVENT_ANNOTATIONS_PATH = DATASET_DIR / "annotations.json"
OPENTTGAMES_DIR        = DATASET_DIR / "openttgames"

WINDOW_SIZE  = 9
FRAME_WIDTH  = 1920.0
FRAME_HEIGHT = 1080.0
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

    cfg = TCNEventNetConfig(
        window_size=WINDOW_SIZE,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )

    own_samples = build_samples(
        ball_store=ball_store,
        event_store=event_store,
        window_size=WINDOW_SIZE,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )
    print(f"Own samples: {len(own_samples)}")

    # ── Optional: merge OpenTTGames annotations ────────────────────────────
    extra_samples = []
    if OPENTTGAMES_DIR.exists():
        extra_samples = build_openttgames_samples(
            data_dir=OPENTTGAMES_DIR,
            window_size=WINDOW_SIZE,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
        )
    else:
        print(
            f"OpenTTGames not found at {OPENTTGAMES_DIR}.\n"
            "  Run: uv run python download_openttgames.py\n"
            "  to download annotation files and improve bounce/net accuracy."
        )

    samples = own_samples + extra_samples
    print(f"Total samples: {len(samples)}")
    counts = {IDX_TO_LABEL[i]: sum(1 for s in samples if s.label_idx == i) for i in range(NUM_CLASSES)}
    print(f"Label distribution: {counts}")

    if len(samples) < 4:
        raise RuntimeError("Not enough samples to train. Annotate more events first.")

    total = len(samples)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * max(counts[IDX_TO_LABEL[i]], 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    )
    print(f"Class weights: { {IDX_TO_LABEL[i]: f'{class_weights[i]:.2f}' for i in range(NUM_CLASSES)} }")

    dataset  = KinematicEventDataset(samples)
    n_val    = max(1, int(len(dataset) * VAL_RATIO))
    n_train  = len(dataset) - n_val
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
