"""Train YOLOv8n-seg on OpenTTGames segmentation masks (table + person).

Pipeline:
  1. Auto-download OpenTTGames training videos + segmentation masks if needed
  2. Convert PNG masks → YOLO polygon labels
  3. Train YOLOv8n-seg (pretrained backbone)
  4. Save best weights and a metrics graph into weights/

OpenTTGames mask encoding (channel-wise):
    B = table, G = person, R = scoreboard (ignored)
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

from download_openttgames import TRAINING_GAMES, download_annotations, download_videos

OPENTTGAMES_DIR = Path("dataset/openttgames")
DATASET_DIR     = Path("seg_dataset")
WEIGHTS_DIR     = Path("weights")

BASE_MODEL        = "yolov8n-seg.pt"
RUN_NAME          = "seg"
BEST_WEIGHTS_NAME = "seg_best.pt"
METRICS_PLOT_NAME = "seg_metrics.png"

CLASS_NAMES  = ["table", "person"]
TABLE_ID     = 0
PERSON_ID    = 1

EPOCHS        = 100
BATCH_SIZE    = 8
IMAGE_SIZE    = 640
VAL_RATIO     = 0.2
SEED          = 42
MIN_MASK_AREA = 64.0


class SegConfig(BaseModel):
    dataset_yaml: Path
    base_model: str
    epochs: int
    batch_size: int
    image_size: int
    weights_dir: Path
    run_name: str


def detect_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_label_file(label_path: Path, lines: list[str]) -> None:
    label_path.write_text("\n".join(lines) + "\n")


def _normalize_polygon(contour: np.ndarray, width: int, height: int) -> str | None:
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return None
    coords = [f"{x / width:.6f} {y / height:.6f}" for x, y in pts]
    return " ".join(coords)


def _mask_to_yolo_lines(mask: np.ndarray, class_id: int) -> list[str]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape
    lines: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_MASK_AREA:
            continue
        polygon = _normalize_polygon(contour, width, height)
        if polygon is None:
            continue
        lines.append(f"{class_id} {polygon}")
    return lines


def _read_mask(mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (table_mask, person_mask) from OpenTTGames PNG."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    if mask.ndim == 3:
        bgr = mask[:, :, :3]
        return (bgr[:, :, 0] > 0).astype(np.uint8), (bgr[:, :, 1] > 0).astype(np.uint8)

    return (mask == 1).astype(np.uint8), (mask == 2).astype(np.uint8)


def _build_game_samples(
    game_dir: Path,
    images_dir: Path,
    labels_dir: Path,
) -> int:
    video_path = game_dir / f"{game_dir.name}.mp4"
    if not video_path.exists():
        raise RuntimeError(f"Missing video: {video_path}")

    mask_paths = sorted(p for p in game_dir.glob("*.png") if p.stem.isdigit())
    if not mask_paths:
        raise RuntimeError(f"No segmentation masks in {game_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    written = 0
    for mask_path in mask_paths:
        frame_idx = int(mask_path.stem)
        table_mask, person_mask = _read_mask(mask_path)

        lines = [
            *_mask_to_yolo_lines(table_mask, TABLE_ID),
            *_mask_to_yolo_lines(person_mask, PERSON_ID),
        ]
        if not lines:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        stem = f"{game_dir.name}_{frame_idx:06d}"
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _write_label_file(labels_dir / f"{stem}.txt", lines)
        written += 1

    cap.release()
    return written


def _split_and_move(all_images_dir: Path, dataset_dir: Path, val_ratio: float, seed: int) -> None:
    images = sorted(all_images_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError("Dataset is empty.")

    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_ratio))
    splits: dict[str, list[Path]] = {"val": images[:n_val], "train": images[n_val:]}

    all_labels_dir = dataset_dir / "_all" / "labels"
    for split, paths in splits.items():
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            shutil.move(str(src), img_dir / src.name)
            lbl = all_labels_dir / f"{src.stem}.txt"
            shutil.move(str(lbl), lbl_dir / lbl.name)

    print(f"Dataset: {len(images)} total | {len(splits['train'])} train / {len(splits['val'])} val")


def _write_yaml(dataset_dir: Path) -> Path:
    yaml_path = dataset_dir / "dataset.yaml"
    names = ", ".join(f"'{n}'" for n in CLASS_NAMES)
    yaml_path.write_text(
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n"
    )
    return yaml_path


def build_dataset(dataset_dir: Path, val_ratio: float, seed: int) -> Path:
    _reset_dir(dataset_dir)
    images_dir = dataset_dir / "_all" / "images"
    labels_dir = dataset_dir / "_all" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for game in TRAINING_GAMES:
        count = _build_game_samples(OPENTTGAMES_DIR / game, images_dir, labels_dir)
        print(f"  {game}: {count} samples")
        total += count

    print(f"Total: {total} samples")
    _split_and_move(images_dir, dataset_dir, val_ratio=val_ratio, seed=seed)
    return _write_yaml(dataset_dir)


def _load_csv(path: Path) -> dict[str, list[float]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {path}")
        fieldnames = [f.strip() for f in reader.fieldnames]
        data: dict[str, list[float]] = {f: [] for f in fieldnames}
        for row in reader:
            for k, v in row.items():
                if k and v and v.strip():
                    data[k.strip()].append(float(v.strip()))
    return data


def plot_metrics(results_csv: Path, output_path: Path) -> None:
    data = _load_csv(results_csv)
    epochs = data.get("epoch")
    if not epochs:
        raise RuntimeError(f"No epoch column in {results_csv}")

    loss_cols   = [c for c in data if (c.startswith("train/") or c.startswith("val/")) and c.endswith("loss")]
    metric_cols = [c for c in data if c.startswith("metrics/")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col in loss_cols:
        axes[0].plot(epochs, data[col], label=col)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    if loss_cols:
        axes[0].legend(fontsize=8)

    for col in metric_cols:
        axes[1].plot(epochs, data[col], label=col)
    axes[1].set_title("Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    if metric_cols:
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train(config: SegConfig) -> Path:
    config.weights_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.base_model)
    model.train(
        data=str(config.dataset_yaml),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.image_size,
        device=detect_device(),
        project=str(config.weights_dir),
        name=config.run_name,
        exist_ok=True,
        save=True,
        plots=True,
    )
    return config.weights_dir / config.run_name


def entrypoint() -> None:
    print("=== Downloading OpenTTGames videos + masks ===")
    failed = download_annotations(TRAINING_GAMES, include_masks=True) + download_videos(TRAINING_GAMES)
    if failed:
        raise RuntimeError(f"Download failed: {', '.join(failed)}")

    print("\n=== Building YOLO dataset ===")
    yaml_path = build_dataset(DATASET_DIR, val_ratio=VAL_RATIO, seed=SEED)

    config = SegConfig(
        dataset_yaml=yaml_path,
        base_model=BASE_MODEL,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        weights_dir=WEIGHTS_DIR,
        run_name=RUN_NAME,
    )

    print("\n=== Training ===")
    run_dir = train(config)

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dest = WEIGHTS_DIR / BEST_WEIGHTS_NAME
        shutil.copy2(best, dest)
        print(f"Best weights → {dest}")

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        plot_path = WEIGHTS_DIR / METRICS_PLOT_NAME
        plot_metrics(results_csv, plot_path)
        print(f"Metrics → {plot_path}")


if __name__ == "__main__":
    entrypoint()
