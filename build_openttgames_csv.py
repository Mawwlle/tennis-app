"""Build a flat CSV from OpenTTGames annotations.

Output columns:
    x_ball,y_ball,table_polygon,grid_bbox,event,filename

OpenTTGames itself contains ball coordinates, event labels, and semantic table
masks. It does not contain a dedicated net/grid mask, so grid_bbox intentionally
stores only the net/grid x coordinate derived from the table-mask center.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import NamedTuple


OPENTTGAMES_DIR = Path("dataset/openttgames")
OUTPUT_CSV = Path("dataset/openttgames_labels.csv")

FRAME_W = 1920
FRAME_H = 1080

CSV_COLUMNS = ["x_ball", "y_ball", "table_polygon", "grid_bbox", "event", "filename"]


class TableGeometry(NamedTuple):
    polygon: tuple[tuple[int, int], ...]
    grid_x: int


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _mask_paths(game_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for mask_dir_name in ("segmentation_masks", "masks", "seg_masks"):
        mask_dir = game_dir / mask_dir_name
        if mask_dir.exists():
            candidates.extend(p for p in mask_dir.glob("*.png") if p.stem.isdigit())
    candidates.extend(p for p in game_dir.glob("*.png") if p.stem.isdigit())
    return sorted(set(candidates), key=lambda p: int(p.stem))


def _scale_polygon(
    polygon: tuple[tuple[int, int], ...],
    mask_width: int,
    mask_height: int,
) -> tuple[tuple[int, int], ...]:
    if mask_width == FRAME_W and mask_height == FRAME_H:
        return polygon

    sx = FRAME_W / mask_width
    sy = FRAME_H / mask_height
    return tuple(
        (
            max(0, min(FRAME_W - 1, int(round(x * sx)))),
            max(0, min(FRAME_H - 1, int(round(y * sy)))),
        )
        for x, y in polygon
    )


def _table_geometry_from_mask(mask_path: Path) -> TableGeometry | None:
    """Return table polygon and center x from one OpenTTGames PNG mask.

    OpenTTGames masks are channel encoded. The local training pipeline treats
    the file RGB R channel as table; cv2 reads PNGs as BGR, hence channel 2.
    """
    import cv2
    import numpy as np

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None

    if mask.ndim == 3:
        table_mask = mask[:, :, 2] > 0
    else:
        table_mask = mask == 1

    binary = table_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    perimeter = cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon=0.006 * perimeter, closed=True)
    points = approx.reshape(-1, 2)
    if len(points) < 3:
        return None

    h, w = table_mask.shape
    polygon = _scale_polygon(
        tuple((int(x), int(y)) for x, y in points),
        mask_width=w,
        mask_height=h,
    )
    xs = [x for x, _ in polygon]
    grid_x = int(round((min(xs) + max(xs)) * 0.5))
    return TableGeometry(polygon=polygon, grid_x=grid_x)


def _table_geometries_from_masks(game_dir: Path) -> tuple[dict[int, TableGeometry], TableGeometry | None]:
    """Return per-frame table geometry and a median fallback for the game."""
    by_frame: dict[int, TableGeometry] = {}
    for mask_path in _mask_paths(game_dir):
        geometry = _table_geometry_from_mask(mask_path)
        if geometry is not None:
            by_frame[int(mask_path.stem)] = geometry

    if not by_frame:
        return by_frame, None

    values = list(by_frame.values())
    median_grid_x = int(round(median(g.grid_x for g in values)))
    representative = min(values, key=lambda geometry: abs(geometry.grid_x - median_grid_x))
    return by_frame, TableGeometry(polygon=representative.polygon, grid_x=median_grid_x)


def _polygon_csv_value(polygon: tuple[tuple[int, int], ...] | None) -> str:
    if polygon is None:
        return ""
    return json.dumps([list(point) for point in polygon], separators=(",", ":"))


def _nearest_geometry(
    frame_idx: int,
    by_frame: dict[int, TableGeometry],
    fallback: TableGeometry | None,
) -> TableGeometry | None:
    if frame_idx in by_frame:
        return by_frame[frame_idx]
    if not by_frame:
        return fallback

    nearest_frame = min(by_frame, key=lambda idx: abs(idx - frame_idx))
    if abs(nearest_frame - frame_idx) <= 30:
        return by_frame[nearest_frame]
    return fallback


def _find_video(game_dir: Path) -> Path | None:
    for suffix in (".mp4", ".mov", ".avi"):
        path = game_dir / f"{game_dir.name}{suffix}"
        if path.exists():
            return path
    videos = sorted(
        p for p in game_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi"}
    )
    return videos[0] if videos else None


def _filename(game_dir: Path, frame_idx: int) -> str:
    video = _find_video(game_dir)
    if video is not None:
        return f"{game_dir.name}/{video.name}#frame={frame_idx}"
    return f"{game_dir.name}/{frame_idx:06d}.jpg"


def _game_dirs(root: Path, only: set[str] | None) -> list[Path]:
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if only is None:
        return dirs
    return [p for p in dirs if p.name in only]


def build_csv(
    data_dir: Path,
    output: Path,
    fallback_grid_x: int,
    only_games: set[str] | None,
    limit_rows: int | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for game_dir in _game_dirs(data_dir, only_games):
            ball_path = game_dir / "ball_markup.json"
            events_path = game_dir / "events_markup.json"
            if not ball_path.exists() or not events_path.exists():
                continue

            table_by_frame, median_table = _table_geometries_from_masks(game_dir)
            if median_table is None:
                print(f"  [warn] {game_dir.name}: no table masks, using fallback grid x={fallback_grid_x}")
            else:
                print(f"  [masks] {game_dir.name}: {len(table_by_frame)} table masks, median grid x={median_table.grid_x}")

            ball_markup = _read_json(ball_path)
            events = {int(k): str(v) for k, v in _read_json(events_path).items()}

            for frame_str, coords_obj in sorted(ball_markup.items(), key=lambda item: int(item[0])):
                coords = coords_obj if isinstance(coords_obj, dict) else {}
                frame_idx = int(frame_str)
                x = int(coords.get("x", -1))
                y = int(coords.get("y", -1))
                geometry = _nearest_geometry(frame_idx, table_by_frame, median_table)
                writer.writerow({
                    "x_ball": x,
                    "y_ball": y,
                    "table_polygon": _polygon_csv_value(geometry.polygon if geometry is not None else None),
                    "grid_bbox": geometry.grid_x if geometry is not None else fallback_grid_x,
                    "event": events.get(frame_idx, "none"),
                    "filename": _filename(game_dir, frame_idx),
                })
                rows_written += 1
                if limit_rows is not None and rows_written >= limit_rows:
                    print(f"Wrote {rows_written} rows -> {output}")
                    return

    print(f"Wrote {rows_written} rows -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenTTGames CSV annotations.")
    parser.add_argument("--data-dir", type=Path, default=OPENTTGAMES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--fallback-grid-x",
        type=int,
        default=FRAME_W // 2,
        help="Net/grid x coordinate used only when a game has no table-mask PNGs.",
    )
    parser.add_argument("--games", nargs="*", default=None, help="Optional subset, e.g. game_1 test_2.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Smoke-test limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    only_games = set(args.games) if args.games else None
    build_csv(
        data_dir=args.data_dir,
        output=args.output,
        fallback_grid_x=args.fallback_grid_x,
        only_games=only_games,
        limit_rows=args.limit_rows,
    )


if __name__ == "__main__":
    main()
