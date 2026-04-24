"""Compute stable table/person masks and net geometry from YOLO-seg detections.

New model classes: table=0, person=1.
Net geometry is derived analytically from the table mask centerline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from eventnet.features import NetGeometry

try:
    from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


def load_net_model(weights: Path) -> "YOLO":
    if not _YOLO_AVAILABLE:
        raise RuntimeError("ultralytics is not installed")
    if not weights.exists():
        raise FileNotFoundError(f"Net segmentation weights not found: {weights}")
    return YOLO(str(weights))


@dataclass
class SegmentationMaps:
    """Averaged masks from the first N frames of the segmentation model."""

    table_mask: np.ndarray | None
    person_mask: np.ndarray | None
    net_geometry: NetGeometry


def _label_from_result(result: object, det_idx: int) -> str:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        return ""

    cls_tensor = boxes.cls[det_idx]
    cls_idx = int(cls_tensor.item())
    if isinstance(names, dict):
        return str(names.get(cls_idx, cls_idx)).lower()
    if isinstance(names, list) and 0 <= cls_idx < len(names):
        return str(names[cls_idx]).lower()
    return str(cls_idx)


def _net_geometry_from_table(table_mask: np.ndarray, target_w: int, target_h: int) -> NetGeometry:
    """Derive net geometry from the table mask centerline (vertical midpoint)."""
    _, xs = np.where(table_mask > 0)
    if len(xs) == 0:
        return NetGeometry()
    x_min, x_max = int(xs.min()), int(xs.max())
    cx_px = (x_min + x_max) // 2
    col = table_mask[:, cx_px]
    col_ys = np.where(col > 0)[0]
    top_y = int(col_ys.min()) if len(col_ys) > 0 else 0
    return NetGeometry(cx=cx_px / target_w, top_y=top_y / target_h)


def compute_segmentation_maps(
    video_path: Path,
    net_model: "YOLO",
    n_frames: int = 30,
    target_w: int = 640,
    target_h: int = 360,
    conf: float = 0.25,
    mask_threshold: float = 0.35,
) -> SegmentationMaps:
    """Run YOLO-seg on the first N frames and return averaged table/person masks.

    New model classes: table=0, person=1.
    Net geometry is derived analytically from the table mask centerline.
    """
    cap = cv2.VideoCapture(str(video_path))

    table_acc  = np.zeros((target_h, target_w), dtype=np.float32)
    person_acc = np.zeros((target_h, target_w), dtype=np.float32)
    table_hits  = 0
    person_hits = 0
    frame_idx = 0

    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        resized: np.ndarray = cv2.resize(frame, (target_w, target_h))
        result = net_model(resized, conf=conf, verbose=False)[0]

        masks = getattr(result, "masks", None)
        if masks is not None:
            for det_idx, mask_tensor in enumerate(masks.data):
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                binary = (mask_resized > 0.5).astype(np.float32)
                label = _label_from_result(result, det_idx)

                if label in {"0", "table"}:
                    table_acc += binary
                    table_hits += 1
                elif label in {"1", "person"}:
                    person_acc += binary
                    person_hits += 1

        frame_idx += 1

    cap.release()

    table_mask  = None
    person_mask = None
    geometry    = NetGeometry()

    if table_hits > 0:
        table_mask = (table_acc / table_hits >= mask_threshold).astype(np.uint8)
        geometry = _net_geometry_from_table(table_mask, target_w, target_h)
        print(f"  [seg] Table mask from {table_hits} detections")
    else:
        print("  [seg] No table detections in first frames — using default geometry")

    if person_hits > 0:
        person_mask = (person_acc / person_hits >= mask_threshold).astype(np.uint8)
        print(f"  [seg] Person mask from {person_hits} detections")

    return SegmentationMaps(
        table_mask=table_mask,
        person_mask=person_mask,
        net_geometry=geometry,
    )


def compute_net_geometry(
    video_path: Path,
    net_model: "YOLO",
    n_frames: int = 30,
    target_w: int = 640,
    target_h: int = 360,
    conf: float = 0.25,
) -> NetGeometry:
    """Backward-compatible helper returning only the net geometry."""
    maps = compute_segmentation_maps(
        video_path=video_path,
        net_model=net_model,
        n_frames=n_frames,
        target_w=target_w,
        target_h=target_h,
        conf=conf,
    )
    return maps.net_geometry
