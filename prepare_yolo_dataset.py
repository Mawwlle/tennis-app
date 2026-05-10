"""Prepare a YOLO ball-detection dataset from TrackNet point annotations.

The script writes:
  dataset/yolo_ball/images/{train,val}/*.jpg
  dataset/yolo_ball/labels/{train,val}/*.txt
  dataset/yolo_ball/dataset.yaml
  dataset/yolo_ball/preview_grid.jpg
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tracknet.dataset import TARGET_H, TARGET_W, Sample
from tracknet.openttgames import build_openttgames_samples

ANNOTATIONS_PATH = Path("dataset/ball_annotations.json")
VIDEOS_DIR = Path("dataset/videos")
OPENTTGAMES_DIR = Path("dataset/openttgames")
DATASET_DIR = Path("dataset/yolo_ball")

VAL_RATIO = 0.2
BOX_SIZE = 18.0
PREVIEW_NAME = "preview_grid.jpg"


@dataclass(frozen=True)
class BuildStats:
    train: int
    val: int
    skipped: int


def _video_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot read video size: {path}")
    return width, height


def load_own_samples(annotations_path: Path, videos_dir: Path) -> list[Sample]:
    data = json.loads(annotations_path.read_text())
    samples: list[Sample] = []
    for video_id, annotations in data["videos"].items():
        video_path = videos_dir / video_id
        if not video_path.exists():
            print(f"  [skip] video not found: {video_path}")
            continue
        orig_w, orig_h = _video_size(video_path)
        for ann in annotations:
            samples.append(
                Sample(
                    video_path=video_path,
                    frame_idx=int(ann["frame_idx"]),
                    cx=float(ann["cx"]),
                    cy=float(ann["cy"]),
                    orig_w=orig_w,
                    orig_h=orig_h,
                    visibility=int(ann["visibility"]),
                )
            )
    return samples


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _slug_video_path(path: Path) -> str:
    stem = f"{path.parent.name}_{path.stem}"
    return "".join(ch if ch.isalnum() else "_" for ch in stem)


def _image_label_paths(dataset_dir: Path, split: str, sample: Sample) -> tuple[Path, Path]:
    stem = f"{_slug_video_path(sample.video_path)}_{sample.frame_idx:06d}"
    image_path = dataset_dir / "images" / split / f"{stem}.jpg"
    label_path = dataset_dir / "labels" / split / f"{stem}.txt"
    return image_path, label_path


def _yolo_line(sample: Sample, box_size: float) -> str:
    cx = sample.cx * TARGET_W / sample.orig_w
    cy = sample.cy * TARGET_H / sample.orig_h
    x = min(1.0, max(0.0, cx / TARGET_W))
    y = min(1.0, max(0.0, cy / TARGET_H))
    w = min(1.0, box_size / TARGET_W)
    h = min(1.0, box_size / TARGET_H)
    return f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def _dedupe_samples(samples: list[Sample]) -> list[Sample]:
    by_key: dict[tuple[Path, int], Sample] = {}
    for sample in samples:
        by_key[(sample.video_path, sample.frame_idx)] = sample
    return list(by_key.values())


def _split_samples(samples: list[Sample], val_ratio: float) -> dict[str, list[Sample]]:
    by_video: dict[Path, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_video[sample.video_path].append(sample)

    train: list[Sample] = []
    val: list[Sample] = []
    for video_samples in by_video.values():
        video_samples.sort(key=lambda s: s.frame_idx)
        cut = max(1, int(len(video_samples) * (1.0 - val_ratio)))
        train.extend(video_samples[:cut])
        val.extend(video_samples[cut:])
    return {"train": train, "val": val}


def write_dataset_yaml(dataset_dir: Path) -> Path:
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names: ['ball']\n"
    )
    return yaml_path


def _write_split(
    dataset_dir: Path,
    split: str,
    samples: list[Sample],
    box_size: float,
    include_empty: bool,
    limit: int | None,
) -> tuple[int, int]:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    by_video: dict[Path, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.visibility == 1 or include_empty:
            by_video[sample.video_path].append(sample)

    if limit is not None:
        limited: dict[Path, list[Sample]] = defaultdict(list)
        remaining = limit
        for video_path, video_samples in by_video.items():
            if remaining <= 0:
                break
            take = video_samples[:remaining]
            limited[video_path].extend(take)
            remaining -= len(take)
        by_video = limited

    total = sum(len(v) for v in by_video.values())
    with tqdm(total=total, desc=f"YOLO {split}", unit="frame") as bar:
        for video_path, video_samples in by_video.items():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                skipped += len(video_samples)
                bar.update(len(video_samples))
                continue
            for sample in sorted(video_samples, key=lambda s: s.frame_idx):
                image_path, label_path = _image_label_paths(dataset_dir, split, sample)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_idx)
                ok, frame = cap.read()
                if not ok:
                    skipped += 1
                    bar.update()
                    continue

                frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
                label_path.write_text(_yolo_line(sample, box_size) + "\n" if sample.visibility == 1 else "")
                written += 1
                bar.update()
            cap.release()
    return written, skipped


def _label_path_for_image(dataset_dir: Path, image_path: Path) -> Path:
    split = image_path.parent.name
    return dataset_dir / "labels" / split / f"{image_path.stem}.txt"


def _draw_yolo_boxes(image: np.ndarray, label_path: Path) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()
    if not label_path.exists():
        return out

    for raw_line in label_path.read_text().splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        _, xc_s, yc_s, bw_s, bh_s = parts
        xc = float(xc_s) * w
        yc = float(yc_s) * h
        bw = float(bw_s) * w
        bh = float(bh_s) * h
        x1 = int(round(xc - bw / 2))
        y1 = int(round(yc - bh / 2))
        x2 = int(round(xc + bw / 2))
        y2 = int(round(yc + bh / 2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (int(round(xc)), int(round(yc))), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def draw_preview_grid(
    dataset_dir: Path,
    output_path: Path | None = None,
    split: str = "train",
    count: int = 16,
) -> Path:
    output_path = output_path or dataset_dir / PREVIEW_NAME
    image_paths = sorted((dataset_dir / "images" / split).glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No preview images found in {dataset_dir / 'images' / split}")

    if len(image_paths) > count:
        idxs = np.linspace(0, len(image_paths) - 1, count, dtype=int)
        image_paths = [image_paths[int(i)] for i in idxs]
    else:
        image_paths = image_paths[:count]

    tile_w, tile_h = 320, 180
    grid_cols = 4
    grid_rows = 4
    canvas = np.zeros((grid_rows * tile_h, grid_cols * tile_w, 3), dtype=np.uint8)

    for i, image_path in enumerate(image_paths[: grid_cols * grid_rows]):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxed = _draw_yolo_boxes(image, _label_path_for_image(dataset_dir, image_path))
        tile = cv2.resize(boxed, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        row = i // grid_cols
        col = i % grid_cols
        y0 = row * tile_h
        x0 = col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
        cv2.putText(
            canvas,
            image_path.stem[-18:],
            (x0 + 6, y0 + tile_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"Preview grid -> {output_path}")
    return output_path


def build_dataset(
    samples: list[Sample],
    dataset_dir: Path,
    val_ratio: float,
    box_size: float,
    include_empty: bool,
    rebuild: bool,
    limit_samples: int | None,
) -> Path:
    yaml_path = dataset_dir / "dataset.yaml"
    if yaml_path.exists() and not rebuild:
        print(f"Dataset already exists: {dataset_dir} (use --rebuild to recreate)")
        draw_preview_grid(dataset_dir)
        return yaml_path

    _reset_dir(dataset_dir)
    samples = _dedupe_samples(samples)
    split_samples = _split_samples(samples, val_ratio)

    train_written, train_skipped = _write_split(
        dataset_dir, "train", split_samples["train"], box_size, include_empty, limit_samples
    )
    val_written, val_skipped = _write_split(
        dataset_dir, "val", split_samples["val"], box_size, include_empty, limit_samples
    )
    stats = BuildStats(train=train_written, val=val_written, skipped=train_skipped + val_skipped)
    if stats.train == 0 or stats.val == 0:
        raise RuntimeError(f"YOLO dataset is too small: {stats}")

    print(f"Dataset: {stats.train} train / {stats.val} val, skipped={stats.skipped}")
    yaml_path = write_dataset_yaml(dataset_dir)
    draw_preview_grid(dataset_dir)
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO ball dataset from TrackNet annotations.")
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--openttgames-dir", type=Path, default=OPENTTGAMES_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--box-size", type=float, default=BOX_SIZE)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--include-empty", action="store_true", help="Include invisible-ball frames as empty labels.")
    parser.add_argument("--no-openttgames", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None, help="Small smoke-test limit per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview_only:
        draw_preview_grid(args.dataset_dir)
        return

    samples = load_own_samples(args.annotations, args.videos_dir)
    print(f"Own samples: {len(samples)}")

    if not args.no_openttgames and args.openttgames_dir.exists():
        samples += build_openttgames_samples(args.openttgames_dir)
    elif args.no_openttgames:
        print("OpenTTGames disabled")
    else:
        print(f"OpenTTGames not found: {args.openttgames_dir}")

    dataset_yaml = build_dataset(
        samples=samples,
        dataset_dir=args.dataset_dir,
        val_ratio=args.val_ratio,
        box_size=args.box_size,
        include_empty=args.include_empty,
        rebuild=args.rebuild,
        limit_samples=args.limit_samples,
    )
    print(f"Prepared YOLO dataset: {dataset_yaml}")


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    main()
