"""Visualize OpenTTGames CSV annotations on exactly one selected video.

The overlay uses only CSV data:
  - table_polygon: table contour polygon
  - grid_bbox: net/grid x coordinate or bbox
  - event: bounce/net/empty_event label

No masks, models, or image analysis are used during visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2


CSV_PATH = Path("dataset/openttgames_labels.csv")
DATA_DIR = Path("dataset/openttgames")
OUTPUT_DIR = Path("visualizations")

COLOR_TABLE = (0, 220, 120)
COLOR_GRID = (0, 165, 255)
COLOR_EVENT = {
    "bounce": (0, 255, 255),
    "net": (0, 80, 255),
    "empty_event": (190, 190, 190),
}
EVENT_HOLD_FRAMES = 18


@dataclass(frozen=True)
class CsvAnnotation:
    frame_idx: int
    table_polygon: tuple[tuple[int, int], ...] | None
    grid_bbox: int | tuple[int, int, int, int] | None
    event: str


@dataclass(frozen=True)
class VideoChoice:
    key: str
    path: Path
    rows: int
    annotated_frames: int
    width: int
    height: int
    fps: float
    frame_count: int
    can_open: bool

    @property
    def duration_s(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def _parse_filename(value: str) -> tuple[str, int] | None:
    if "#frame=" not in value:
        return None
    video_part, frame_part = value.rsplit("#frame=", 1)
    try:
        return video_part, int(frame_part)
    except ValueError:
        return None


def _parse_table_bbox(value: str) -> tuple[int, int, int, int] | None:
    value = value.strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != 4:
        return None
    return tuple(int(round(float(v))) for v in parsed)  # type: ignore[return-value]


def _parse_table_polygon(
    polygon_value: str,
    bbox_value: str = "",
) -> tuple[tuple[int, int], ...] | None:
    polygon_value = polygon_value.strip()
    if polygon_value:
        try:
            parsed = json.loads(polygon_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and len(parsed) >= 3:
            points: list[tuple[int, int]] = []
            for point in parsed:
                if not isinstance(point, list) or len(point) != 2:
                    return None
                points.append((int(round(float(point[0]))), int(round(float(point[1])))))
            return tuple(points)

    # Backward compatibility with old CSVs that still contain table_bbox.
    bbox = _parse_table_bbox(bbox_value)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _parse_grid_bbox(value: str) -> int | tuple[int, int, int, int] | None:
    value = value.strip()
    if not value:
        return None
    if value.startswith("["):
        return _parse_table_bbox(value)
    try:
        return int(round(float(value)))
    except ValueError:
        return None


def _read_csv_by_video(csv_path: Path) -> dict[str, dict[int, CsvAnnotation]]:
    by_video: dict[str, dict[int, CsvAnnotation]] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = _parse_filename(row.get("filename", ""))
            if parsed is None:
                continue
            video_key, frame_idx = parsed
            by_video.setdefault(video_key, {})[frame_idx] = CsvAnnotation(
                frame_idx=frame_idx,
                table_polygon=_parse_table_polygon(
                    row.get("table_polygon", ""),
                    row.get("table_bbox", ""),
                ),
                grid_bbox=_parse_grid_bbox(row.get("grid_bbox", "")),
                event=row.get("event", "none").strip() or "none",
            )
    return by_video


def _video_metadata(path: Path) -> tuple[bool, int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, 0, 0, 0.0, 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width > 0 and height > 0 and fps > 0, width, height, fps, frame_count


def _build_choices(
    data_dir: Path,
    by_video: dict[str, dict[int, CsvAnnotation]],
) -> list[VideoChoice]:
    choices: list[VideoChoice] = []
    for video_key, annotations in sorted(by_video.items()):
        path = data_dir / video_key
        can_open, width, height, fps, frame_count = _video_metadata(path)
        choices.append(VideoChoice(
            key=video_key,
            path=path,
            rows=len(annotations),
            annotated_frames=len(annotations),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            can_open=can_open,
        ))
    return choices


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def _print_menu(choices: list[VideoChoice]) -> None:
    print("\nVideos from CSV:")
    for idx, choice in enumerate(choices, start=1):
        if choice.can_open:
            meta = (
                f"{choice.width}x{choice.height}, "
                f"{choice.frame_count} frames, "
                f"{choice.fps:.2f} fps, "
                f"{_format_duration(choice.duration_s)}"
            )
        else:
            meta = "cannot open video"
        print(f"  {idx:2d}. {choice.key} | {meta} | csv rows={choice.rows}")


def _select_choice(choices: list[VideoChoice], video_index: int | None) -> VideoChoice:
    if not choices:
        raise RuntimeError("No videos found in CSV.")

    if video_index is not None:
        if video_index < 1 or video_index > len(choices):
            raise ValueError(f"--video-index must be between 1 and {len(choices)}")
        choice = choices[video_index - 1]
        if not choice.can_open:
            raise RuntimeError(f"Selected video cannot be opened: {choice.path}")
        return choice

    _print_menu(choices)
    while True:
        raw = input("\nSelect one video number: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if idx < 1 or idx > len(choices):
            print(f"Choose a number from 1 to {len(choices)}.")
            continue
        choice = choices[idx - 1]
        if not choice.can_open:
            print(f"Cannot open this video: {choice.path}")
            continue
        return choice


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    sx = frame_w / 1920.0
    sy = frame_h / 1080.0
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def _scale_polygon(
    polygon: tuple[tuple[int, int], ...],
    frame_w: int,
    frame_h: int,
) -> tuple[tuple[int, int], ...]:
    sx = frame_w / 1920.0
    sy = frame_h / 1080.0
    return tuple((int(round(x * sx)), int(round(y * sy))) for x, y in polygon)


def _scale_x(x: int, frame_w: int) -> int:
    return int(round(x * frame_w / 1920.0))


def _draw_table(frame, polygon: tuple[tuple[int, int], ...] | None) -> None:
    if polygon is None or len(polygon) < 3:
        return

    import numpy as np

    scaled = _scale_polygon(polygon, frame.shape[1], frame.shape[0])
    points = np.array(scaled, dtype=np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], COLOR_TABLE, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
    cv2.polylines(frame, [points], isClosed=True, color=COLOR_TABLE, thickness=2, lineType=cv2.LINE_AA)


def _draw_grid(frame, grid: int | tuple[int, int, int, int] | None) -> None:
    if grid is None:
        return
    h, w = frame.shape[:2]
    if isinstance(grid, tuple):
        x1, y1, x2, y2 = _scale_bbox(grid, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GRID, 3, cv2.LINE_AA)
        return
    x = _scale_x(grid, w)
    cv2.line(frame, (x, 0), (x, h - 1), COLOR_GRID, 3, cv2.LINE_AA)


def _draw_event(frame, event: str, frame_idx: int) -> None:
    if event == "none":
        return
    color = COLOR_EVENT.get(event, (255, 255, 255))
    label = f"{event}  frame {frame_idx}"
    cv2.rectangle(frame, (24, 24), (390, 76), (0, 0, 0), -1)
    cv2.rectangle(frame, (24, 24), (390, 76), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (40, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _nearest_annotation(
    frame_idx: int,
    annotations: dict[int, CsvAnnotation],
    sorted_frames: list[int],
) -> CsvAnnotation | None:
    if frame_idx in annotations:
        return annotations[frame_idx]
    if not sorted_frames:
        return None

    # The CSV is sparse; use the closest geometry row so the table/grid remain visible.
    nearest = min(sorted_frames, key=lambda idx: abs(idx - frame_idx))
    return annotations[nearest]


def _active_event(
    frame_idx: int,
    event_frames: list[int],
    annotations: dict[int, CsvAnnotation],
) -> tuple[str, int] | None:
    if not event_frames:
        return None
    nearest = min(event_frames, key=lambda idx: abs(idx - frame_idx))
    if abs(nearest - frame_idx) <= EVENT_HOLD_FRAMES:
        return annotations[nearest].event, nearest
    return None


def render_video(
    choice: VideoChoice,
    annotations: dict[int, CsvAnnotation],
    output: Path,
    max_frames: int | None,
) -> Path:
    cap = cv2.VideoCapture(str(choice.path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {choice.path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter.fourcc(*"mp4v"),
        choice.fps,
        (choice.width, choice.height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer: {output}")

    sorted_frames = sorted(annotations)
    event_frames = sorted(idx for idx, ann in annotations.items() if ann.event != "none")
    total = choice.frame_count if max_frames is None else min(choice.frame_count, max_frames)

    frame_idx = 0
    while frame_idx < total:
        ok, frame = cap.read()
        if not ok:
            break

        ann = _nearest_annotation(frame_idx, annotations, sorted_frames)
        if ann is not None:
            _draw_table(frame, ann.table_polygon)
            _draw_grid(frame, ann.grid_bbox)

        active = _active_event(frame_idx, event_frames, annotations)
        if active is not None:
            event, event_frame = active
            _draw_event(frame, event, event_frame)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Rendered {frame_idx} frames -> {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one OpenTTGames CSV/video pair.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--video-index", type=int, default=None, help="1-based menu index; skips interactive input.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_video = _read_csv_by_video(args.csv)
    choices = _build_choices(args.data_dir, by_video)
    _print_menu(choices)
    choice = _select_choice(choices, args.video_index)

    output = args.output
    if output is None:
        stem = choice.key.replace("/", "_").rsplit(".", 1)[0]
        output = OUTPUT_DIR / f"{stem}_csv_vis.mp4"

    render_video(
        choice=choice,
        annotations=by_video[choice.key],
        output=output,
        max_frames=args.max_frames,
    )


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    main()
