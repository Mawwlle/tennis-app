"""Score a video using YOLO ball detection.

This is the YOLO-only analogue of score.py:
  Pass 1  YOLO ball detector -> ball positions + interpolation
          + trajectory heuristics -> hit / bounce / net / miss events
          + rally state machine -> score timeline
  Pass 2  Render annotated MP4 with score overlay + audio

Usage:
    uv run python score_yolo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from score import (
    NET_CX_RATIO,
    OUTPUT,
    TARGET_H,
    TARGET_W,
    TRAIL_WINDOW,
    VIDEO_PATH,
    FrameScore,
    _attach_score_audio,
    _draw_ball,
    _draw_bounce_marker,
    _draw_hit_marker,
    _draw_miss_marker,
    _draw_net_line,
    _draw_net_marker,
    _draw_score_bar,
    _draw_trail,
    _filter_hits_near_bounces,
    _interpolate,
    build_score_timeline,
    build_trajectory_series,
    detect_bounces,
    detect_hits,
    detect_hits_from_bounces,
    detect_miss_events,
    detect_net_events,
    merge_events,
)

if TYPE_CHECKING:
    from ultralytics import YOLO


YOLO_WEIGHTS = Path("weights/yolo_det.pt")
YOLO_OUTPUT = OUTPUT.with_name("score_yolo_result.mp4")
YOLO_CONF_THRESHOLD = 0.85
YOLO_INFER_STEP = 1


def _load_yolo(weights: Path) -> "YOLO":
    from ultralytics import YOLO  # type: ignore[reportMissingImports]

    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}. "
            "Train first with: uv run train-yolo --rebuild"
        )
    return YOLO(str(weights))


def _predict_ball(
    model: "YOLO",
    frame_bgr: NDArray[np.uint8],
    conf: float,
) -> tuple[float, float, float] | None:
    result = model(frame_bgr, conf=conf, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    names = getattr(result, "names", {}) or {}
    best: tuple[float, float, float] | None = None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    has_ball_class = any(str(v).lower() == "ball" for v in names.values()) if isinstance(names, dict) else False

    for box, score, cls_idx in zip(xyxy, confs, classes):
        label = str(names.get(int(cls_idx), cls_idx)).lower() if isinstance(names, dict) else str(cls_idx)
        if has_ball_class and label != "ball":
            continue

        x1, y1, x2, y2 = box
        cx = float((x1 + x2) * 0.5)
        cy = float((y1 + y2) * 0.5)
        candidate = (cx, cy, float(score))
        if best is None or candidate[2] > best[2]:
            best = candidate

    return best


def run(video_path: Path, output: Path, weights: Path = YOLO_WEIGHTS) -> None:
    print(f"YOLO weights: {weights}")
    model = _load_yolo(weights)
    net_cx_px = NET_CX_RATIO * TARGET_W

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0 or fps <= 0:
        raise RuntimeError(f"Cannot open video metadata: {video_path}")

    print(f"Pass 1 — YOLO on {total} frames …")
    detections: dict[int, tuple[float, float]] = {}
    detection_scores: dict[int, float] = {}

    for fi in tqdm(range(total), desc="YOLO", unit="frame"):
        ret, raw = cap.read()
        if not ret:
            break
        if fi % YOLO_INFER_STEP != 0:
            continue

        frame = cv2.resize(raw, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        pred = _predict_ball(model, frame, YOLO_CONF_THRESHOLD)
        if pred is None:
            continue
        cx, cy, score = pred
        detections[fi] = (cx, cy)
        detection_scores[fi] = score
    cap.release()

    all_positions = _interpolate(detections)
    print(f"  detected {len(detections)} / interpolated {len(all_positions)} frames")
    if detection_scores:
        scores = np.array(list(detection_scores.values()), dtype=np.float32)
        print(f"  confidence: min={scores.min():.3f} mean={scores.mean():.3f} max={scores.max():.3f}")

    print("Detecting events …")
    traj = build_trajectory_series(all_positions)
    if traj is None:
        raise RuntimeError("Not enough YOLO tracked points to score the video.")

    bounces = detect_bounces(traj, net_cx_px)
    hits = detect_hits_from_bounces(bounces, all_positions, net_cx_px)
    if not hits:
        hits = detect_hits(traj, net_cx_px)
    hits = _filter_hits_near_bounces(hits, bounces)
    nets = detect_net_events(traj, hits, bounces, net_cx_px)
    misses = detect_miss_events(traj, hits, bounces, net_cx_px)
    events = merge_events(hits, bounces, nets, misses)

    print(f"  hits: {len(hits)}  bounces: {len(bounces)}  nets: {len(nets)}  misses: {len(misses)}")
    for ev in events:
        print(f"    frame {ev.frame_idx:5d}  {ev.kind:6s}  side={ev.side:5s}  cx={ev.cx:6.1f}  cy={ev.cy:6.1f}")

    timeline = build_score_timeline(events, detections, total)
    final = timeline.get(total - 1, FrameScore())
    print(f"  Final score: LEFT {final.left} : RIGHT {final.right}")

    print("Pass 2 — rendering …")
    silent_output = output.with_name(f"{output.stem}.silent.mp4")
    writer = cv2.VideoWriter(
        str(silent_output),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (TARGET_W, TARGET_H),
    )

    known_frames = sorted(detections)
    cap = cv2.VideoCapture(str(video_path))
    for fi in tqdm(range(total), desc="Rendering", unit="frame"):
        ret, raw = cap.read()
        if not ret:
            break

        frame = cv2.resize(raw, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        trail_keys = [i for i in known_frames if i <= fi][-TRAIL_WINDOW:]
        _draw_trail(frame, [detections[i] for i in trail_keys])

        if fi in detections:
            _draw_ball(frame, *detections[fi], detected=True)
        elif fi in all_positions:
            _draw_ball(frame, *all_positions[fi], detected=False)

        for bounce in bounces:
            _draw_bounce_marker(frame, bounce, fi)
        for hit in hits:
            _draw_hit_marker(frame, hit, fi)
        for net in nets:
            _draw_net_marker(frame, net, fi)
        for miss in misses:
            _draw_miss_marker(frame, miss, fi)

        _draw_net_line(frame, net_cx_px)
        _draw_score_bar(frame, timeline.get(fi, FrameScore()))
        writer.write(frame)

    cap.release()
    writer.release()
    _attach_score_audio(silent_output, output, timeline, fps, total)
    print(f"Saved -> {output}")


def main() -> None:
    run(VIDEO_PATH, YOLO_OUTPUT)


if __name__ == "__main__":
    main()
