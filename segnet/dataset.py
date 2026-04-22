"""Dataset for SegNet: reads YOLO-seg polygon labels → pixel masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# YOLO class_id → mask value  (0 = background)
# classes.txt: 0=net, 1=table  →  mask: 1=net, 2=table
YOLO_TO_MASK: dict[int, int] = {0: 1, 1: 2}


def _load_mask(label_path: Path, height: int, width: int) -> np.ndarray:
    """Parse YOLO-seg label file into an (H, W) uint8 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)

    text = label_path.read_text().strip()
    if not text:
        return mask

    for line in text.splitlines():
        parts = line.split()
        class_id = int(parts[0])
        mask_value = YOLO_TO_MASK.get(class_id)
        if mask_value is None:
            continue

        coords = list(map(float, parts[1:]))
        xs = [round(coords[i] * width) for i in range(0, len(coords), 2)]
        ys = [round(coords[i] * height) for i in range(1, len(coords), 2)]
        pts = np.array(list(zip(xs, ys)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], mask_value)

    return mask


class SegNetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        image_paths: list[Path],
        label_paths: list[Path],
        image_size: tuple[int, int],  # (H, W)
    ) -> None:
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = self.image_size

        img_bgr = cv2.imread(str(self.image_paths[idx]))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {self.image_paths[idx]}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = _load_mask(self.label_paths[idx], h, w)

        image_tensor = self.to_tensor(img_resized)
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


def build_dataset(
    images_dir: Path,
    labels_dir: Path,
    image_size: tuple[int, int],
) -> SegNetDataset:
    image_paths = sorted(images_dir.glob("*.jpg"))
    label_paths = [labels_dir / p.with_suffix(".txt").name for p in image_paths]

    # Keep only pairs where label exists and is non-empty
    pairs = [
        (img, lbl)
        for img, lbl in zip(image_paths, label_paths)
        if lbl.exists() and lbl.stat().st_size > 0
    ]

    imgs, lbls = zip(*pairs) if pairs else ([], [])
    return SegNetDataset(list(imgs), list(lbls), image_size)
