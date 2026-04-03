"""Train TrackNet on ball_annotations.json.

If dataset/openttgames/ exists and contains video files, OpenTTGames samples
are merged with the manually-annotated dataset before training.

To download OpenTTGames videos first::

    uv run python download_openttgames.py --video
"""

import json
from pathlib import Path

import cv2
import torch

from tracknet.dataset import Sample, build_loaders
from tracknet.openttgames import build_openttgames_samples
from tracknet.train import run_training

ANNOTATIONS_PATH = Path("dataset/ball_annotations.json")
DATASET_DIR      = Path("dataset/videos")
OPENTTGAMES_DIR  = Path("dataset/openttgames")
OUTPUT_DIR       = Path("weights")

EPOCHS     = 100
BATCH_SIZE = 4
LR         = 1.0       # Adadelta default
VAL_RATIO  = 0.2


def _video_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def load_own_samples(annotations_path: Path, dataset_dir: Path) -> list[Sample]:
    data = json.loads(annotations_path.read_text())
    samples: list[Sample] = []
    for video_id, anns in data["videos"].items():
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

    own_samples = load_own_samples(ANNOTATIONS_PATH, DATASET_DIR)
    print(f"Own samples: {len(own_samples)} annotated frames from "
          f"{len(set(s.video_path for s in own_samples))} videos")

    # ── Optional: merge OpenTTGames samples ───────────────────────────────
    extra_samples: list[Sample] = []
    if OPENTTGAMES_DIR.exists():
        extra_samples = build_openttgames_samples(OPENTTGAMES_DIR)
    else:
        print(
            f"OpenTTGames not found at {OPENTTGAMES_DIR}.\n"
            "  Run: uv run python download_openttgames.py --video\n"
            "  to download videos and improve TrackNet accuracy."
        )

    samples = own_samples + extra_samples
    if extra_samples:
        print(f"Total samples: {len(samples)} "
              f"(own={len(own_samples)}, openttgames={len(extra_samples)})")

    train_loader, val_loader = build_loaders(samples, VAL_RATIO, BATCH_SIZE)

    run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        output_dir=OUTPUT_DIR,
    )


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    main()
