"""Full pipeline: TrackNet → interpolation → net segmentation → EventNet.

Pass 0: YOLO-seg on first 30 frames → average net geometry
Pass 1: TrackNet every INFER_STEP frames → spline interpolation of ball positions
Pass 2: EventNet on sliding 15-frame window → game state per frame
Pass 3: Render side-by-side MP4
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

from eventnet.dataset import IDX_TO_LABEL
from eventnet.features import NetGeometry
from eventnet.heatmap import positions_to_heatmaps
from eventnet.model import HeatmapEventNet, HeatmapEventNetConfig
from eventnet.segmentation import compute_net_geometry, load_net_model
from tracknet.model import TrackNet

VIDEO_PATH       = Path("test_2.mp4")
WEIGHTS          = Path("weights/tracknet_best.pt")
EVENTNET_WEIGHTS = Path("weights/eventnet_best.pt")
NET_SEG_WEIGHTS  = Path("weights/seg_best.pt")
OUTPUT           = Path("infer_result_new.mp4")

TARGET_W       = 640
TARGET_H       = 360
CONF_THRESHOLD = 0.99
TRAIL_WINDOW   = 9
INFER_STEP     = 3
SEG_FRAMES     = 30   # frames used to estimate net geometry


_EVENT_COLORS: dict[str, tuple[int, int, int]] = {
    "hit":    (0,   215, 255),
    "bounce": (255, 220, 0),
    "net":    (0,   60,  220),
    "none":   (120, 120, 120),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_tracknet(weights: Path, device: torch.device) -> TrackNet:
    model = TrackNet()
    model.load_state_dict(torch.load(str(weights), map_location=device, weights_only=True))
    return model.to(device).eval()


def load_eventnet(
    weights: Path,
    device: torch.device,
) -> tuple[HeatmapEventNet, HeatmapEventNetConfig] | None:
    if not weights.exists():
        return None
    checkpoint = torch.load(str(weights), map_location=device, weights_only=True)
    cfg = HeatmapEventNetConfig(**checkpoint["cfg"])
    model = HeatmapEventNet(cfg).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model, cfg


# ---------------------------------------------------------------------------
# Pass 0: net geometry via YOLO segmentation
# ---------------------------------------------------------------------------


def estimate_net_geometry(video_path: Path) -> NetGeometry:
    """Segment first SEG_FRAMES frames with YOLO-seg and return mean net geometry."""
    if not NET_SEG_WEIGHTS.exists():
        print("Net seg weights not found — using default geometry")
        return NetGeometry()

    try:
        net_model = load_net_model(NET_SEG_WEIGHTS)
        print(f"Estimating net geometry from first {SEG_FRAMES} frames…")
        return compute_net_geometry(
            video_path=video_path,
            net_model=net_model,
            n_frames=SEG_FRAMES,
            target_w=TARGET_W,
            target_h=TARGET_H,
        )
    except Exception as exc:
        print(f"  [seg] Failed ({exc}) — using default geometry")
        return NetGeometry()


# ---------------------------------------------------------------------------
# Pass 1: TrackNet detection + interpolation
# ---------------------------------------------------------------------------


def predict(
    model: TrackNet,
    frames: list[NDArray[np.float32]],
    device: torch.device,
) -> tuple[float, float, float, NDArray[np.float32]]:
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


def interpolate_positions(
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


# ---------------------------------------------------------------------------
# Pass 2: EventNet classification
# ---------------------------------------------------------------------------


def classify_events(
    model: HeatmapEventNet,
    cfg: HeatmapEventNetConfig,
    all_positions: dict[int, tuple[float, float]],
    device: torch.device,
) -> dict[int, tuple[str, float]]:
    """Classify game events using HeatmapEventNet over a sliding window of Gaussian heatmaps."""
    half = cfg.window_size // 2
    all_frames = sorted(all_positions)
    results: dict[int, tuple[str, float]] = {}

    for center in all_frames:
        positions: tuple[tuple[float, float] | None, ...] = tuple(
            (all_positions[idx][0] / TARGET_W, all_positions[idx][1] / TARGET_H)
            if idx in all_positions else None
            for idx in range(center - half, center + half + 1)
        )
        heatmaps = positions_to_heatmaps(positions)                     # (W, HM_H, HM_W)
        tensor = torch.from_numpy(heatmaps).unsqueeze(0).to(device)    # (1, W, HM_H, HM_W)

        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[0]

        label_idx = int(probs.argmax().item())
        results[center] = (IDX_TO_LABEL[label_idx], float(probs[label_idx].item()))

    return results


# ---------------------------------------------------------------------------
# Pass 3: rendering
# ---------------------------------------------------------------------------


def _draw_event_label(
    frame: NDArray[np.uint8],
    label: str,
    confidence: float,
) -> None:
    color = _EVENT_COLORS[label]
    text = f"{label}  {confidence:.0%}"
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


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


def _to_heatmap(heatmap_np: NDArray[np.float32]) -> np.ndarray:
    hm_u8 = (heatmap_np * 255).astype(np.uint8)
    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_INFERNO)  # type: ignore[return-value]


def _draw_net_line(frame: NDArray[np.uint8], geo: NetGeometry) -> None:
    """Draw the estimated net position as a vertical line."""
    x = int(geo.cx * TARGET_W)
    y = int(geo.top_y * TARGET_H)
    cv2.line(frame, (x, y), (x, TARGET_H), (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(weights: Path, video_path: Path, output: Path) -> None:
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")

    tracknet = load_tracknet(weights, device)

    event_loaded = load_eventnet(EVENTNET_WEIGHTS, device)
    if event_loaded is None:
        print("EventNet weights not found — skipping event classification")
    else:
        print("EventNet loaded")

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # ── Pass 0: net geometry ──────────────────────────────────────────────────
    net_geometry = estimate_net_geometry(video_path)

    # ── Pass 1: detect + interpolate ──────────────────────────────────────────
    print(f"Pass 1 — TrackNet on {total} frames…")
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
            cx, cy, _, _ = predict(tracknet, frame_buffer[-3:], device)
            if cx >= 0:
                detections[frame_idx] = (cx, cy)
    cap.release()

    print(f"  detected {len(detections)}/{total} frames")
    all_positions = interpolate_positions(detections)
    print(f"  after interpolation: {len(all_positions)} frames")

    # ── Pass 2: event classification ──────────────────────────────────────────
    event_labels: dict[int, tuple[str, float]] = {}
    if event_loaded is not None:
        print("Pass 2 — EventNet classification…")
        event_model, event_cfg = event_loaded
        event_labels = classify_events(
            model=event_model,
            cfg=event_cfg,
            all_positions=all_positions,
            device=device,
        )

    # ── Pass 3: render ────────────────────────────────────────────────────────
    print("Pass 3 — rendering…")
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
            _, _, _, heatmap = predict(tracknet, frame_buffer[-3:], device)

        left: np.ndarray = cv2.resize(raw, (TARGET_W, TARGET_H))

        trail_keys = [i for i in known_frames if i <= frame_idx][-TRAIL_WINDOW:]
        trail = [detections[i] for i in trail_keys]
        _draw_spline_trail(left, trail)

        if frame_idx in detections:
            cx, cy = detections[frame_idx]
            _draw_ball(left, cx, cy, detected=True)
        elif frame_idx in all_positions:
            cx, cy = all_positions[frame_idx]
            _draw_ball(left, cx, cy, detected=False)

        if frame_idx in event_labels:
            label, conf = event_labels[frame_idx]
            _draw_event_label(left, label, conf)

        _draw_net_line(left, net_geometry)

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
