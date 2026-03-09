from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel

from tracknet.model import TrackNet

WEIGHTS_PATH = Path("weights/tracknet_best.pt")
TARGET_W = 640
TARGET_H = 360
HEATMAP_THRESHOLD = 0.4

_model: TrackNet | None = None
_device: torch.device | None = None


def _get_model() -> tuple[TrackNet, torch.device]:
    global _model, _device
    if _model is None:
        _device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
        m = TrackNet()
        m.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=_device, weights_only=True))
        m.to(_device)
        m.eval()
        _model = m
    return _model, _device  # type: ignore[return-value]


class TrackNetResult(BaseModel):
    cx: float
    cy: float
    conf: float


def _read_frame_small(video_path: Path, frame_idx: int) -> NDArray[np.float32]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return np.zeros((TARGET_H, TARGET_W, 3), dtype=np.float32)
    small = cv2.resize(frame, (TARGET_W, TARGET_H)).astype(np.float32) / 255.0
    return small


def detect_all_frames_batch(
    video_path: Path,
    orig_w: int,
    orig_h: int,
) -> list[tuple[int, TrackNetResult]]:
    """Run TrackNet on all frames sequentially, opening the video once."""
    model, device = _get_model()

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_buffer: list[NDArray[np.float32]] = []
    results: list[tuple[int, TrackNetResult]] = []

    for frame_idx in range(total):
        ret, raw = cap.read()
        if not ret:
            break

        small = cv2.resize(raw, (TARGET_W, TARGET_H)).astype(np.float32) / 255.0
        frame_buffer.append(small)
        if len(frame_buffer) > 3:
            frame_buffer.pop(0)

        frames = frame_buffer.copy()
        while len(frames) < 3:
            frames.insert(0, frames[0])

        stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            heatmap = torch.sigmoid(logits)[0, 0]

        heatmap_np: NDArray[np.float32] = heatmap.cpu().numpy()
        confidence = float(heatmap_np.max())

        if confidence >= HEATMAP_THRESHOLD:
            flat = int(heatmap_np.argmax())
            cy_small, cx_small = divmod(flat, TARGET_W)
            cx = cx_small * orig_w / TARGET_W
            cy = cy_small * orig_h / TARGET_H
            results.append((frame_idx, TrackNetResult(cx=cx, cy=cy, conf=confidence)))

    cap.release()
    return results


def detect_ball_tracknet(
    video_path: Path,
    frame_idx: int,
    orig_w: int,
    orig_h: int,
) -> TrackNetResult | None:
    """Run TrackNet on frames [t-2, t-1, t] and return ball position in original coords."""
    model, device = _get_model()

    frames = [
        _read_frame_small(video_path, frame_idx - 2),
        _read_frame_small(video_path, frame_idx - 1),
        _read_frame_small(video_path, frame_idx),
    ]

    stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        heatmap = torch.sigmoid(logits)[0, 0]

    heatmap_np: NDArray[np.float32] = heatmap.cpu().numpy()
    confidence = float(heatmap_np.max())

    if confidence < HEATMAP_THRESHOLD:
        return None

    flat = int(heatmap_np.argmax())
    cy_small, cx_small = divmod(flat, TARGET_W)

    cx = cx_small * orig_w / TARGET_W
    cy = cy_small * orig_h / TARGET_H

    return TrackNetResult(cx=cx, cy=cy, conf=confidence)