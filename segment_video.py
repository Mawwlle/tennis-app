"""Segment tennis video: table + person via YOLO-seg.

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

VIDEO_PATH = Path("test_7.mp4")
NET_WEIGHTS = Path("yolo_seg.pt")
OUTPUT = Path("seg_result.mp4")

TARGET_W = 640
TARGET_H = 360
NET_CONF = 0.25

# BGR colours
COLOR_PERSON: tuple[int, int, int] = (0, 255, 0)    # green
COLOR_TABLE:  tuple[int, int, int] = (0, 128, 255)  # orange
OVERLAY_ALPHA = 0.40

def load_net_model(weights: Path) -> YOLO:
    return YOLO(str(weights))


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


def predict_masks(
    model: YOLO,
    frame_bgr: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns (person_masks, table_masks) as uint8 arrays.

    New model classes: table=0, person=1.
    """
    result = model(frame_bgr, conf=NET_CONF, verbose=False)[0]
    person_masks: list[np.ndarray] = []
    table_masks: list[np.ndarray] = []

    if result.masks is None:
        return person_masks, table_masks

    h, w = frame_bgr.shape[:2]
    for det_idx, mask_tensor in enumerate(result.masks.data):
        mask_np = mask_tensor.cpu().numpy()
        resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (resized > 0.5).astype(np.uint8)
        label = _label_from_result(result, det_idx)
        if label in {"0", "table"}:
            table_masks.append(binary)
        elif label in {"1", "person"}:
            person_masks.append(binary)

    return person_masks, table_masks


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def draw_overlay(
    frame: np.ndarray,
    person_masks: list[np.ndarray],
    table_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (overlay_frame, mask_canvas)."""
    out = frame.copy()
    blend = out.copy()
    canvas = np.zeros_like(frame)

    for table_mask in table_masks:
        blend[table_mask == 1] = COLOR_TABLE
        canvas[table_mask == 1] = COLOR_TABLE
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_TABLE, 2)

    for person_mask in person_masks:
        blend[person_mask == 1] = COLOR_PERSON
        canvas[person_mask == 1] = COLOR_PERSON
        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_PERSON, 2)

    cv2.addWeighted(blend, OVERLAY_ALPHA, out, 1 - OVERLAY_ALPHA, 0, out)
    return out, canvas


def draw_legend(frame: np.ndarray) -> None:
    for i, (label, color) in enumerate([("table", COLOR_TABLE), ("person", COLOR_PERSON)]):
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

        net_masks, table_masks = predict_masks(net_model, frame)

        left, right = draw_overlay(frame, net_masks, table_masks)
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
