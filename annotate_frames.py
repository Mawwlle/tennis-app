"""
Annotate tennis video frames with net/table segmentation.

Pipeline:
  1. Extract every N-th frame from test_2.mp4
  2. Net:   upload to Synth, call segment_objects(prompt="net")
     Table: HSV color thresholding (ping-pong tables are blue)
  3. Save annotations in YOLO-seg and COCO formats
  4. Build a 4x4 segmentation matrix of 16 frames for verification
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel
from synth_client import BoundingBox, SynthClient

VIDEO_PATH = Path("test_2.mp4")
OUTPUT_DIR = Path("annotations_output")
FRAMES_DIR = OUTPUT_DIR / "frames"
LABELS_DIR = OUTPUT_DIR / "labels"
FRAME_STEP = 6
SYNTH_BASE_URL = "http://10.32.11.23"
MATRIX_PATH = OUTPUT_DIR / "verification_matrix.jpg"
CLASSES = ["net", "table"]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class FrameSegmentation(BaseModel):
    frame_idx: int
    image_path: Path
    width: int
    height: int
    boxes: list[BoundingBox]  # each box may have .polygon; .label is "net" or "table"


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    step: int,
) -> list[tuple[int, Path]]:
    """Extract every `step`-th frame. Returns list of (original_frame_idx, saved_path)."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    extracted: list[tuple[int, Path]] = []
    frame_idx = 0

    print(f"Extracting every {step}nd frame from {total} total frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0 and frame is not None:
            path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            extracted.append((frame_idx, path))

        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"  processed {frame_idx}/{total}, extracted {len(extracted)}")

    cap.release()
    print(f"Extracted {len(extracted)} frames to {frames_dir}")
    return extracted


# ---------------------------------------------------------------------------
# Table segmentation — blue HSV thresholding
# ---------------------------------------------------------------------------

# Ping-pong tables are bright blue; tune these if needed
# Sampled table colours (OpenCV HSV, H: 0-180):
#   light  rgb(43,68,184)  → H≈115 S≈195 V≈184
#   light  rgb(32,53,172)  → H≈115 S≈207 V≈172
#   dark   rgb(11,1,152)   → H≈122 S≈253 V≈152
#   dark   rgb(17,0,177)   → H≈123 S≈255 V≈177
#   bg     rgb(9,14,96)    → H≈118 S≈231 V≈96   ← must be excluded
# V_lower=130 separates background (V≈96) from darkest table (V≈152)
TABLE_HSV_LOWER = np.array([105, 60, 130])
TABLE_HSV_UPPER = np.array([130, 255, 255])
# Reject any contour larger than this fraction of the frame (catches background bleed)
TABLE_MAX_AREA_RATIO = 0.50
TABLE_MIN_AREA = 5000  # px² — ignore tiny blue blobs


TABLE_POLY_EPSILON_RATIO = 0.02  # single approxPolyDP pass


def segment_table_by_color(img: np.ndarray) -> list[BoundingBox]:
    """Find the blue table surface via HSV thresholding.

    Keeps only the largest contour that naturally approximates to exactly
    4 corners — no forced simplification of other shapes.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TABLE_HSV_LOWER, TABLE_HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = img.shape[0] * img.shape[1]

    best_cnt: np.ndarray | None = None
    best_quad: list[list[int]] | None = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < TABLE_MIN_AREA:
            continue
        if area / frame_area > TABLE_MAX_AREA_RATIO:
            continue

        epsilon = TABLE_POLY_EPSILON_RATIO * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) != 4:
            continue  # not a natural quadrilateral — skip

        if area > best_area:
            best_area = area
            best_cnt = cnt
            best_quad = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

    if best_cnt is None or best_quad is None:
        return []

    x, y, bw, bh = cv2.boundingRect(best_cnt)
    return [
        BoundingBox(
            label="table",
            x_min=float(x),
            y_min=float(y),
            x_max=float(x + bw),
            y_max=float(y + bh),
            polygon=best_quad,
        )
    ]


# ---------------------------------------------------------------------------
# Segmentation via Synth (net) + color thresholding (table)
# ---------------------------------------------------------------------------


def segment_frame(
    client: SynthClient,
    frame_idx: int,
    image_path: Path,
) -> FrameSegmentation:
    """Upload frame once; segment net via Synth, table via HSV thresholding."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]

    image_dto = client.images.upload(str(image_path))
    net_boxes: list[BoundingBox] = client.images.segment_objects(
        image_id=image_dto.id,
        prompt="net",
    )
    for box in net_boxes:
        box.label = "net"

    table_boxes = segment_table_by_color(img)

    return FrameSegmentation(
        frame_idx=frame_idx,
        image_path=image_path,
        width=w,
        height=h,
        boxes=net_boxes + table_boxes,
    )


def run_segmentations(
    client: SynthClient,
    frames: list[tuple[int, Path]],
) -> list[FrameSegmentation]:
    results: list[FrameSegmentation] = []

    for i, (frame_idx, path) in enumerate(frames):
        seg = segment_frame(client, frame_idx, path)
        results.append(seg)

        if (i + 1) % 50 == 0:
            print(f"  segmented {i + 1}/{len(frames)} frames")

    print(f"Segmentation complete: {len(results)} frames")
    return results


# ---------------------------------------------------------------------------
# YOLO-seg export
# Format: class_id x1 y1 x2 y2 ... xn yn  (all coords normalized 0-1)
# ---------------------------------------------------------------------------


def polygon_to_yolo_seg(
    box: BoundingBox,
    img_w: int,
    img_h: int,
    class_map: dict[str, int],
) -> str | None:
    class_id = class_map.get(box.label.lower())
    if class_id is None:
        return None

    if box.polygon:
        coords = " ".join(
            f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in box.polygon
        )
    else:
        # Fallback: represent bbox as 4-point polygon
        x1, y1 = box.x_min / img_w, box.y_min / img_h
        x2, y2 = box.x_max / img_w, box.y_max / img_h
        coords = f"{x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"

    return f"{class_id} {coords}"


def save_yolo_seg(
    segmentations: list[FrameSegmentation],
    labels_dir: Path,
    class_map: dict[str, int],
) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir.parent / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    for seg in segmentations:
        lines: list[str] = []
        for box in seg.boxes:
            line = polygon_to_yolo_seg(box, seg.width, seg.height, class_map)
            if line is not None:
                lines.append(line)

        label_path = labels_dir / f"frame_{seg.frame_idx:06d}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"YOLO-seg labels saved to {labels_dir}")


# ---------------------------------------------------------------------------
# COCO export (with segmentation polygons)
# ---------------------------------------------------------------------------


def save_coco(
    segmentations: list[FrameSegmentation],
    output_path: Path,
) -> None:
    categories = [
        {"id": i, "name": name, "supercategory": "tennis"}
        for i, name in enumerate(CLASSES)
    ]
    class_map = {name: i for i, name in enumerate(CLASSES)}

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 0

    for seg in segmentations:
        images.append(
            {
                "id": seg.frame_idx,
                "file_name": seg.image_path.name,
                "width": seg.width,
                "height": seg.height,
            }
        )

        for box in seg.boxes:
            cat_id = class_map.get(box.label.lower())
            if cat_id is None:
                continue

            bw = box.x_max - box.x_min
            bh = box.y_max - box.y_min

            if box.polygon:
                segmentation = [[coord for pt in box.polygon for coord in pt]]
            else:
                segmentation = [
                    [box.x_min, box.y_min, box.x_max, box.y_min,
                     box.x_max, box.y_max, box.x_min, box.y_max]
                ]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": seg.frame_idx,
                    "category_id": cat_id,
                    "bbox": [box.x_min, box.y_min, bw, bh],
                    "segmentation": segmentation,
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    output_path.write_text(
        json.dumps(
            {
                "info": {"description": "Tennis net/table segmentation"},
                "categories": categories,
                "images": images,
                "annotations": annotations,
            },
            indent=2,
        )
    )
    print(f"COCO saved to {output_path}  ({ann_id} masks, {len(images)} images)")


# ---------------------------------------------------------------------------
# 4×4 verification matrix
# ---------------------------------------------------------------------------

COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "net": (0, 255, 0),
    "table": (0, 128, 255),
}
DEFAULT_COLOR: tuple[int, int, int] = (255, 0, 0)

THUMB_W = 480
THUMB_H = 270
GRID_COLS = 4
GRID_ROWS = 4


def draw_segmentation(frame: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()

    for box in boxes:
        color = COLOR_MAP.get(box.label.lower(), DEFAULT_COLOR)

        if box.polygon:
            pts = np.array(box.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        else:
            cv2.rectangle(
                overlay,
                (int(box.x_min), int(box.y_min)),
                (int(box.x_max), int(box.y_max)),
                color,
                -1,
            )
            cv2.rectangle(
                out,
                (int(box.x_min), int(box.y_min)),
                (int(box.x_max), int(box.y_max)),
                color,
                2,
            )

        cv2.putText(
            out,
            box.label,
            (int(box.x_min), max(int(box.y_min) - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    return out


def build_verification_matrix(
    segmentations: list[FrameSegmentation],
    output_path: Path,
    n: int = 16,
) -> None:
    """Sample n frames evenly and arrange as a 4×4 grid."""
    total = len(segmentations)
    indices = [int(i * total / n) for i in range(n)]
    sampled = [segmentations[i] for i in indices]

    thumbs: list[np.ndarray] = []
    for seg in sampled:
        frame = cv2.imread(str(seg.image_path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {seg.image_path}")
        annotated = draw_segmentation(frame, seg.boxes)
        thumb = cv2.resize(annotated, (THUMB_W, THUMB_H))

        caption = f"frame {seg.frame_idx} | {len(seg.boxes)} masks"
        cv2.putText(
            thumb, caption, (6, THUMB_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        thumbs.append(thumb)

    rows = [
        np.concatenate(thumbs[r * GRID_COLS : (r + 1) * GRID_COLS], axis=1)
        for r in range(GRID_ROWS)
    ]
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Verification matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(VIDEO_PATH, FRAMES_DIR, FRAME_STEP)
    class_map = {name: i for i, name in enumerate(CLASSES)}

    with SynthClient(base_url=SYNTH_BASE_URL) as client:
        segmentations = run_segmentations(client, frames)

    save_yolo_seg(segmentations, LABELS_DIR, class_map)
    save_coco(segmentations, OUTPUT_DIR / "annotations_coco.json")
    build_verification_matrix(segmentations, MATRIX_PATH)

    print("\nDone.")
    print(f"  Frames:          {FRAMES_DIR}")
    print(f"  YOLO-seg labels: {LABELS_DIR}")
    print(f"  COCO JSON:       {OUTPUT_DIR / 'annotations_coco.json'}")
    print(f"  Matrix:          {MATRIX_PATH}")


if __name__ == "__main__":
    main()
