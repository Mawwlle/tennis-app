"""Train YOLOv8-seg on net segmentation.

Reads YOLO-seg annotations from annotations_output/, filters to net class only,
prepares a proper train/val split, then trains yolov8n-seg.

Usage:
    uv run train-seg
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from pydantic import BaseModel
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

ANNOTATIONS_DIR = Path("annotations_output")
FRAMES_DIR = ANNOTATIONS_DIR / "frames"
LABELS_DIR = ANNOTATIONS_DIR / "labels"
DATASET_DIR = Path("seg_dataset")
WEIGHTS_DIR = Path("weights")

# class_id in the source annotations that corresponds to "net"
NET_CLASS_ID = 0

BASE_MODEL = "yolov8n-seg.pt"
EPOCHS = 100
BATCH_SIZE = 8
IMAGE_SIZE = 640
VAL_RATIO = 0.2
SEED = 42


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class NetSegConfig(BaseModel):
    dataset_yaml: Path
    base_model: str
    epochs: int
    batch_size: int
    image_size: int
    weights_dir: Path


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _filter_net_lines(label_path: Path) -> list[str]:
    """Keep only net lines (class_id == NET_CLASS_ID), remapped to class 0."""
    lines = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if int(parts[0]) == NET_CLASS_ID:
            lines.append("0 " + " ".join(parts[1:]))
    return lines


def build_yolo_dataset(
    frames_dir: Path,
    labels_dir: Path,
    dataset_dir: Path,
    val_ratio: float,
    seed: int,
) -> Path:
    """Build train/val split with net-only labels. Returns dataset.yaml path."""
    # Keep only frames that have at least one net annotation
    candidates: list[Path] = []
    for img in sorted(frames_dir.glob("*.jpg")):
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists() and _filter_net_lines(lbl):
            candidates.append(img)

    rng = random.Random(seed)
    rng.shuffle(candidates)

    n_val = max(1, int(len(candidates) * val_ratio))
    splits: dict[str, list[Path]] = {
        "val":   candidates[:n_val],
        "train": candidates[n_val:],
    }

    print(f"Net frames: {len(candidates)} total  |  {len(splits['train'])} train / {len(splits['val'])} val")

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split, frames in splits.items():
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in frames:
            shutil.copy2(img, img_dir / img.name)
            lbl = labels_dir / f"{img.stem}.txt"
            net_lines = _filter_net_lines(lbl)
            (lbl_dir / f"{img.stem}.txt").write_text("\n".join(net_lines) + "\n")

    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['net']\n"
    )
    return yaml_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(config: NetSegConfig) -> None:
    config.weights_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.base_model)
    model.train(
        data=str(config.dataset_yaml),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.image_size,
        device=detect_device(),
        project=str(config.weights_dir),
        name="net_seg",
        exist_ok=True,
        save=True,
        plots=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def entrypoint() -> None:
    yaml_path = build_yolo_dataset(
        frames_dir=FRAMES_DIR,
        labels_dir=LABELS_DIR,
        dataset_dir=DATASET_DIR,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    config = NetSegConfig(
        dataset_yaml=yaml_path,
        base_model=BASE_MODEL,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        weights_dir=WEIGHTS_DIR,
    )

    train(config)

    best = WEIGHTS_DIR / "net_seg" / "weights" / "best.pt"
    if best.exists():
        dest = WEIGHTS_DIR / "net_seg_best.pt"
        shutil.copy2(best, dest)
        print(f"Best weights → {dest}")


if __name__ == "__main__":
    entrypoint()
