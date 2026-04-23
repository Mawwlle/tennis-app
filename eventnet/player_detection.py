"""Estimate approximate player/racket positions from video using YOLOv8 person detection.

Uses the pretrained COCO yolov8n model (class 0 = person).  Racket position is
approximated as the *lower-centre* of each player bounding box (hands are roughly
at 75% of bbox height in table-tennis stance).

The result is averaged over the first N frames to produce a stable `PlayerGeometry`
that can be injected into EventNet features.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel


class PlayerGeometry(BaseModel):
    """Normalised (x, y) of the estimated racket position for each player.

    Defaults represent a typical fixed-camera table-tennis angle where the left
    player stands at ~15 % and the right player at ~85 % of frame width.
    """

    left_cx:  float = 0.15
    left_cy:  float = 0.65
    right_cx: float = 0.85
    right_cy: float = 0.65


def compute_player_geometry(
    video_path: Path,
    n_frames: int = 60,
    target_w: int = 640,
    target_h: int = 360,
    conf: float = 0.30,
) -> PlayerGeometry:
    """Run YOLOv8n person detection on the first *n_frames* and average bbox positions.

    Args:
        video_path: Path to the source video.
        n_frames:   How many frames to sample (spread evenly through the video).
        target_w:   Resize width before detection.
        target_h:   Resize height before detection.
        conf:       YOLO confidence threshold.

    Returns:
        PlayerGeometry with normalised (cx, cy) for left and right players.
        Falls back to defaults when no persons are detected on a side.
    """
    from ultralytics import YOLO  # local import — avoids slow startup when not needed

    model = YOLO("yolov8n.pt")   # downloads once, then cached

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_pts:  list[tuple[float, float]] = []
    right_pts: list[tuple[float, float]] = []

    step = max(1, total // n_frames)
    for fi in range(0, min(total, n_frames * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        small   = cv2.resize(frame, (target_w, target_h))
        results = model(small, classes=[0], conf=conf, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / target_w
            # Racket ≈ lower quarter of the player bbox (hands/paddle area)
            cy = (y1 * 0.25 + y2 * 0.75) / target_h
            if cx < 0.5:
                left_pts.append((cx, cy))
            else:
                right_pts.append((cx, cy))

    cap.release()

    def _mean(pts: list[tuple[float, float]], default: tuple[float, float]) -> tuple[float, float]:
        if not pts:
            return default
        arr = np.array(pts, dtype=np.float32)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())

    lx, ly = _mean(left_pts,  (0.15, 0.65))
    rx, ry = _mean(right_pts, (0.85, 0.65))
    return PlayerGeometry(left_cx=lx, left_cy=ly, right_cx=rx, right_cy=ry)
