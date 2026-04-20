"""Fine-tune TrackNet on a subset of videos (e.g., night footage).

Usage:
    uv run python finetune_tracknet.py

By default loads weights/tracknet_best.pt and trains only on videos
whose path contains any of the VIDEO_FILTER patterns.

Saves fine-tuned weights to weights/tracknet_finetuned.pt.
"""

import json
from pathlib import Path

import cv2
import torch

from tracknet.dataset import Sample, build_loaders
from tracknet.train import run_training

ANNOTATIONS_PATH = Path("dataset/ball_annotations.json")
DATASET_DIR      = Path("dataset/videos")
PRETRAINED_PATH  = Path("weights/tracknet_best.pt")
OUTPUT_DIR       = Path("weights")
OUTPUT_WEIGHTS   = "tracknet_finetuned.pt"

# Videos whose path contains any of these strings will be included.
# Set to [] to use all annotated videos.
VIDEO_FILTER = ["night_videos"]

EPOCHS     = 30
BATCH_SIZE = 4
LR         = 0.1    # lower than base training (1.0) to preserve learned features
VAL_RATIO  = 0.2


def _video_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def load_filtered_samples(
    annotations_path: Path,
    dataset_dir: Path,
    video_filter: list[str],
) -> list[Sample]:
    data = json.loads(annotations_path.read_text())
    samples: list[Sample] = []
    for video_id, anns in data["videos"].items():
        if video_filter and not any(f in video_id for f in video_filter):
            continue
        video_path = dataset_dir / video_id
        if not video_path.exists():
            print(f"  WARNING: video not found: {video_path}")
            continue
        orig_w, orig_h = _video_size(video_path)
        for ann in anns:
            samples.append(Sample(
                video_path=video_path,
                frame_idx=ann["frame_idx"],
                cx=ann["cx"],
                cy=ann["cy"],
                orig_w=orig_w,
                orig_h=orig_h,
                visibility=ann["visibility"],
            ))
    return samples


def main() -> None:
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")

    samples = load_filtered_samples(ANNOTATIONS_PATH, DATASET_DIR, VIDEO_FILTER)
    if not samples:
        print("No samples found. Annotate the video via 'uv run webapp' first.")
        return

    filter_desc = ", ".join(VIDEO_FILTER) if VIDEO_FILTER else "all videos"
    print(f"Fine-tune samples: {len(samples)} frames from filter=[{filter_desc}]")

    train_loader, val_loader = build_loaders(samples, VAL_RATIO, BATCH_SIZE)

    best_path = run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        output_dir=OUTPUT_DIR,
        pretrained_path=PRETRAINED_PATH,
    )

    # Rename the output to avoid overwriting the original best weights
    finetuned_path = OUTPUT_DIR / OUTPUT_WEIGHTS
    best_path.rename(finetuned_path)
    print(f"Fine-tuned weights saved → {finetuned_path}")


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    main()
