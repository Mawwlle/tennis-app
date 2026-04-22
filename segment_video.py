"""Segment tennis video: net via YOLO-seg, table via HSV thresholding.

Output: side-by-side MP4 — left: original + coloured overlay,
        right: binary mask canvas.

Usage:
    uv run python segment_video.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

from annotate_frames import segment_table_by_color

VIDEO_PATH = Path("test_2.mp4")
NET_WEIGHTS = Path("weights/seg_best.pt")
OUTPUT = Path("seg_result.mp4")

TARGET_W = 640
TARGET_H = 360
NET_CONF = 0.25

# BGR colours
COLOR_NET:   tuple[int, int, int] = (0, 255, 0)    # green
COLOR_TABLE: tuple[int, int, int] = (0, 128, 255)  # orange-blue
OVERLAY_ALPHA = 0.40

# HSV thresholds copied from annotate_frames
TABLE_HSV_LOWER = np.array([105, 60, 130])
TABLE_HSV_UPPER = np.array([130, 255, 255])


# ---------------------------------------------------------------------------
# Net segmentation — YOLO
# ---------------------------------------------------------------------------


def load_net_model(weights: Path) -> YOLO:
    return YOLO(str(weights))


def predict_net_masks(
    model: YOLO,
    frame_bgr: np.ndarray,
) -> list[np.ndarray]:
    """Returns list of (H, W) uint8 binary masks for each detected net."""
    result = model(frame_bgr, conf=NET_CONF, verbose=False)[0]
    masks: list[np.ndarray] = []

    if result.masks is None:
        return masks

    h, w = frame_bgr.shape[:2]
    for mask_tensor in result.masks.data:
        mask_np = mask_tensor.cpu().numpy()
        resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
        masks.append((resized > 0.5).astype(np.uint8))

    return masks


# ---------------------------------------------------------------------------
# Table segmentation — HSV (largest blue quad)
# ---------------------------------------------------------------------------


def predict_table_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Returns (H, W) uint8 binary mask for the table."""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    boxes = segment_table_by_color(frame_bgr)
    for box in boxes:
        if box.polygon:
            pts = np.array(box.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)

    return mask


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def draw_overlay(
    frame: np.ndarray,
    net_masks: list[np.ndarray],
    table_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (overlay_frame, mask_canvas)."""
    out = frame.copy()
    blend = out.copy()
    canvas = np.zeros_like(frame)

    # Table
    blend[table_mask == 1] = COLOR_TABLE
    canvas[table_mask == 1] = COLOR_TABLE

    contours_t, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours_t, -1, COLOR_TABLE, 2)

    # Net
    for net_mask in net_masks:
        blend[net_mask == 1] = COLOR_NET
        canvas[net_mask == 1] = COLOR_NET
        contours_n, _ = cv2.findContours(net_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours_n, -1, COLOR_NET, 2)

    cv2.addWeighted(blend, OVERLAY_ALPHA, out, 1 - OVERLAY_ALPHA, 0, out)
    return out, canvas


def draw_legend(frame: np.ndarray) -> None:
    for i, (label, color) in enumerate([("net", COLOR_NET), ("table", COLOR_TABLE)]):
        x, y = 10, 24 + i * 24
        cv2.rectangle(frame, (x, y - 14), (x + 16, y), color, -1)
        cv2.putText(frame, label, (x + 22, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(net_weights: Path, video_path: Path, output: Path) -> None:
    net_model = load_net_model(net_weights)

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (TARGET_W * 2, TARGET_H),
    )

    for _ in tqdm(range(total), desc="Segmenting", unit="frame"):
        ret, raw = cap.read()
        if not ret:
            break

        frame: np.ndarray = cv2.resize(raw, (TARGET_W, TARGET_H))

        net_masks   = predict_net_masks(net_model, frame)
        table_mask  = predict_table_mask(frame)

        left, right = draw_overlay(frame, net_masks, table_mask)
        draw_legend(left)

        combined = np.hstack([left, right])
        cv2.line(combined, (TARGET_W, 0), (TARGET_W, TARGET_H - 1), (60, 60, 60), 2)

        writer.write(combined)

    cap.release()
    writer.release()
    print(f"Saved → {output}")


def main() -> None:
    run(NET_WEIGHTS, VIDEO_PATH, OUTPUT)


if __name__ == "__main__":
    main()
