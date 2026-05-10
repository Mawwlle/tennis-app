"""Train YOLO detector on a prepared ball dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

DATASET_YAML = Path("dataset/yolo_ball/dataset.yaml")
WEIGHTS_DIR = Path("weights")
BASE_MODEL = "yolov8n.pt"
RUN_NAME = "yolo_ball"
OUTPUT_WEIGHTS = WEIGHTS_DIR / "yolo_det.pt"

EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640

KNOWN_BASE_MODELS = {
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8n.yaml",
    "yolov8s.yaml",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11n.yaml",
    "yolo11s.yaml",
}
BASE_MODEL_ALIASES = {
    "yolov26s.pt": "yolov8s.pt",
    "yolov26n.pt": "yolov8n.pt",
}


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_base_model(base_model: str) -> str:
    alias = BASE_MODEL_ALIASES.get(base_model)
    if alias is not None:
        print(f"Base model alias: {base_model} -> {alias}")
        return alias

    path = Path(base_model)
    if path.exists():
        return base_model

    if base_model in KNOWN_BASE_MODELS:
        return base_model

    known = ", ".join(sorted(KNOWN_BASE_MODELS))
    raise FileNotFoundError(
        f"Base model not found: {base_model}\n"
        f"Use an existing local .pt file or one of: {known}\n"
        "For offline training from scratch, use: --base-model yolov8n.yaml"
    )


def train_yolo(
    dataset_yaml: Path,
    base_model: str,
    epochs: int,
    batch: int,
    image_size: int,
    device: str,
    run_name: str,
    output_weights: Path,
) -> Path:
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found: {dataset_yaml}\n"
            "Run first: uv run prepare-yolo-dataset"
        )

    model = YOLO(resolve_base_model(base_model))
    project = WEIGHTS_DIR
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=image_size,
        device=device,
        project=str(project),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        single_cls=True,
    )

    best = project / run_name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished, but best weights not found: {best}")
    output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output_weights)
    print(f"Best weights -> {output_weights}")
    return output_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO ball detector on prepared dataset.")
    parser.add_argument("--data", type=Path, default=DATASET_YAML)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-weights", type=Path, default=OUTPUT_WEIGHTS)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device() if args.device == "auto" else args.device
    print(f"Device: {device}")
    train_yolo(
        dataset_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        batch=args.batch,
        image_size=args.imgsz,
        device=device,
        run_name=args.run_name,
        output_weights=args.output_weights,
    )


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    main()
