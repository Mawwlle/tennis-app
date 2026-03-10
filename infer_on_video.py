"""Apply TrackNet to a video. Left: ball tracking + trail. Right: heatmap."""

from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

from tracknet.model import TrackNet

VIDEO_PATH = Path("dataset/videos/sasha_tichka/Screen Recording 2026-02-23 at 16.18.27.mov")
WEIGHTS    = Path("weights/tracknet_best.pt")
OUTPUT     = Path("infer_result_new.mp4")

TARGET_W       = 640
TARGET_H       = 360
CONF_THRESHOLD = 0.98
TRAIL_WINDOW   = 9
INFER_STEP     = 3   # run TrackNet every N frames; gaps filled by interpolation


def load_model(weights: Path, device: torch.device) -> TrackNet:
    model = TrackNet()
    model.load_state_dict(torch.load(str(weights), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict(
    model: TrackNet,
    frames: list[NDArray[np.float32]],
    device: torch.device,
) -> tuple[float, float, float, NDArray[np.float32]]:
    """Returns (cx, cy, confidence, heatmap_np). cx=-1 if below threshold."""
    stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        heatmap = torch.sigmoid(logits)[0, 0]

    heatmap_np: NDArray[np.float32] = heatmap.cpu().numpy()
    confidence = float(heatmap_np.max())

    if confidence < CONF_THRESHOLD:
        return -1.0, -1.0, confidence, heatmap_np

    flat = int(heatmap_np.argmax())
    cy, cx = divmod(flat, TARGET_W)
    return float(cx), float(cy), confidence, heatmap_np


def _interpolate(
    detections: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    if len(detections) < 2:
        return dict(detections)

    known_idx = sorted(detections)
    xs = np.array([detections[i][0] for i in known_idx], dtype=float)
    ys = np.array([detections[i][1] for i in known_idx], dtype=float)

    k = min(3, len(known_idx) - 1)
    spl_x = make_interp_spline(known_idx, xs, k=k)
    spl_y = make_interp_spline(known_idx, ys, k=k)

    first, last = known_idx[0], known_idx[-1]
    return {i: (float(spl_x(i)), float(spl_y(i))) for i in range(first, last + 1)}


def _draw_spline_trail(
    frame: NDArray[np.uint8],
    trail: list[tuple[float, float]],
) -> None:
    if len(trail) < 2:
        return

    if len(trail) >= 4:
        idx = np.arange(len(trail), dtype=float)
        k = min(3, len(trail) - 1)
        spl_x = make_interp_spline(idx, [p[0] for p in trail], k=k)
        spl_y = make_interp_spline(idx, [p[1] for p in trail], k=k)
        t = np.linspace(0, len(trail) - 1, 60)
        curve = [(int(spl_x(v)), int(spl_y(v))) for v in t]
    else:
        curve = [(int(p[0]), int(p[1])) for p in trail]

    for j in range(1, len(curve)):
        alpha = j / len(curve)
        cv2.line(
            frame, curve[j - 1], curve[j],
            (int(255 * alpha), int(220 * alpha), 0),
            max(1, int(alpha * 4)), cv2.LINE_AA,
        )


def _draw_ball(
    frame: NDArray[np.uint8],
    cx: float,
    cy: float,
    detected: bool,
) -> None:
    x, y = int(cx), int(cy)
    if detected:
        cv2.circle(frame, (x, y), 8, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x, y), 6, (180, 180, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1, cv2.LINE_AA)


def _to_heatmap(heatmap_np: NDArray[np.float32]) -> NDArray[np.uint8]:
    hm_u8 = (heatmap_np * 255).astype(np.uint8)
    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_INFERNO)


def run(weights: Path, video_path: Path, output: Path) -> None:
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")
    model = load_model(weights, device)

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # ── Pass 1: detect ────────────────────────────────────────────────────────
    print(f"Pass 1 — running TrackNet on {total} frames…")
    detections: dict[int, tuple[float, float]] = {}
    frame_buffer: list[NDArray[np.float32]] = []

    cap = cv2.VideoCapture(str(video_path))
    for frame_idx in tqdm(range(total), desc="Detecting", unit="frame"):
        ret, raw = cap.read()
        if not ret:
            break

        small = cv2.resize(raw, (TARGET_W, TARGET_H)).astype(np.float32) / 255.0
        frame_buffer.append(small)
        if len(frame_buffer) > 3:
            frame_buffer.pop(0)
        while len(frame_buffer) < 3:
            frame_buffer.insert(0, frame_buffer[0])

        if frame_idx % INFER_STEP == 0:
            cx, cy, _, _ = predict(model, frame_buffer[-3:], device)
            if cx >= 0:
                detections[frame_idx] = (cx, cy)
    cap.release()

    print(f"  detected {len(detections)}/{total} frames")
    all_positions = _interpolate(detections)
    print(f"  after interpolation: {len(all_positions)} frames have a position")

    # ── Pass 2: render ────────────────────────────────────────────────────────
    print("Pass 2 — rendering…")
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (TARGET_W * 2, TARGET_H),
    )

    frame_buffer = []
    known_frames = sorted(detections)
    heatmap = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    for frame_idx in tqdm(range(total), desc="Rendering", unit="frame"):
        ret, raw = cap.read()
        if not ret:
            break

        small = cv2.resize(raw, (TARGET_W, TARGET_H)).astype(np.float32) / 255.0
        frame_buffer.append(small)
        if len(frame_buffer) > 3:
            frame_buffer.pop(0)
        while len(frame_buffer) < 3:
            frame_buffer.insert(0, frame_buffer[0])

        if frame_idx % INFER_STEP == 0:
            _, _, _, heatmap = predict(model, frame_buffer[-3:], device)
        # else: reuse previous heatmap

        # Left: tracking on resized frame
        left = cv2.resize(raw, (TARGET_W, TARGET_H))

        trail_keys = [i for i in known_frames if i <= frame_idx][-TRAIL_WINDOW:]
        trail = [detections[i] for i in trail_keys]
        _draw_spline_trail(left, trail)

        if frame_idx in detections:
            cx, cy = detections[frame_idx]
            _draw_ball(left, cx, cy, detected=True)
        elif frame_idx in all_positions:
            cx, cy = all_positions[frame_idx]
            _draw_ball(left, cx, cy, detected=False)

        # Right: heatmap
        right = _to_heatmap(heatmap)

        combined = np.hstack([left, right])
        cv2.line(combined, (TARGET_W, 0), (TARGET_W, TARGET_H - 1), (60, 60, 60), 2)

        writer.write(combined)

    cap.release()
    writer.release()
    print(f"Saved → {output}")


def main() -> None:
    run(WEIGHTS, VIDEO_PATH, OUTPUT)


if __name__ == "__main__":
    main()
