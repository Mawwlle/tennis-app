"""Score solver: trajectory-only event spotting and scoring.

This pipeline does not use EventNet. All events are inferred from the ball
trajectory produced by TrackNet:

  Pass 0  YOLO-seg (optional)  -> net geometry
  Pass 1  TrackNet + spline    -> ball positions
  Pass 2  trajectory heuristics -> hit / bounce / net events
  Pass 3  rally state machine  -> score timeline
  Pass 4  render annotated MP4

Usage:
    uv run python score.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import tempfile
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ultralytics import YOLO
import wave

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from tqdm import tqdm

from eventnet.features import NetGeometry
from eventnet.segmentation import compute_segmentation_maps, load_net_model
from tracknet.model import TrackNet

VIDEO_PATH        = Path("dataset/videos/normal_point/Screen Recording 2026-02-23 at 16.02.35.mov")
TRACKNET_WEIGHTS  = Path("weights/tracknet_best.pt")
SEG_WEIGHTS       = Path("weights/seg_best.pt")     # unified table+person model
EVENTNET_WEIGHTS  = Path("weights/eventnet_best.pt")
OUTPUT            = Path("score_result.mp4")

EVENTNET_THRESHOLD = 0.50   # sigmoid threshold for EventNet peak detection
EVENTNET_DEDUP     = 8      # min frames between EventNet events

TARGET_W = 640
TARGET_H = 360
CONF_THRESHOLD = 0.99
TRAIL_WINDOW   = 18
INFER_STEP     = 4
SEG_FRAMES     = 1
SEG_INTERVAL   = 100   # re-run segmentation every N frames (players move)

# Trajectory smoothing
TRAJ_SMOOTH_WINDOW = 6
TRAJ_SMOOTH_POLY   = 2

# Event detection heuristics
LOCAL_WINDOW            = 4
HIT_DEDUP_FRAMES        = 10
BOUNCE_DEDUP_FRAMES     = 12
NET_DEDUP_FRAMES        = 12
HIT_MIN_X_SPEED         = 1.4
HIT_MIN_TRAVEL_PX       = 18.0
HIT_NET_MARGIN_RATIO    = 0.10
HIT_Y_MAX_RATIO         = 0.92
BOUNCE_MIN_Y_RATIO      = 0.42
BOUNCE_MAX_Y_RATIO      = 0.90
BOUNCE_MIN_DY           = 1.0
BOUNCE_MIN_CURVATURE    = 1.8
BOUNCE_MIN_X_SPEED      = 0.60
NET_NEAR_RATIO          = 0.07
NET_REVERSAL_X_SPEED    = 0.90
NET_MAX_CROSS_RATIO     = 0.03
NET_MIN_Y_RATIO         = 0.20
NET_MAX_Y_RATIO         = 0.78
NET_POST_HIT_FRAMES     = 90
NET_MIN_PROGRESS_RATIO  = 0.35
MAX_HIT_TO_BOUNCE_FRAMES = 110
MAX_BOUNCE_TO_HIT_FRAMES = 140
BALL_LOST_FRAMES        = 60
EVENT_TEXT_FADE_FRAMES  = 26
MISS_MAX_Y_RATIO        = 0.88
MISS_OUT_X_MARGIN_RATIO = 0.03
MISS_RETURN_X_SPEED     = 0.75
TABLE_MASK_MARGIN_PX    = 10


Side = Literal["left", "right"]
EventKind = Literal["hit", "bounce", "net", "miss"]


class FrameScore(BaseModel):
    left: int = 0
    right: int = 0
    point_winner: Side | None = None
    event_label: str | None = None


class HitEvent(BaseModel):
    frame_idx: int
    side: Side
    cx: float
    cy: float
    confidence: float


class BounceEvent(BaseModel):
    frame_idx: int
    side: Side
    cx: float
    cy: float
    confidence: float


class NetEvent(BaseModel):
    frame_idx: int
    side: Side
    cx: float
    cy: float
    confidence: float


class MissEvent(BaseModel):
    frame_idx: int
    side: Side
    cx: float
    cy: float
    confidence: float


class TrajectoryEvent(BaseModel):
    frame_idx: int
    kind: EventKind
    side: Side
    cx: float
    cy: float
    confidence: float


@dataclass
class TrajectorySeries:
    frames: list[int]
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    x_s: NDArray[np.float32]
    y_s: NDArray[np.float32]
    dx: NDArray[np.float32]
    dy: NDArray[np.float32]
    ddx: NDArray[np.float32]
    ddy: NDArray[np.float32]


@dataclass
class CalibrationData:
    net_mask: np.ndarray
    table_mask: np.ndarray
    net_geometry: NetGeometry


@dataclass
class RallyStateMachine:
    """Finite-state scoring from trajectory events."""

    phase: Literal["idle", "await_bounce", "await_return"] = field(default="idle")
    striker: Side | None = field(default=None)
    target_side: Side | None = field(default=None)
    returner_side: Side | None = field(default=None)
    striker_side: Side | None = field(default=None)
    last_event_frame: int = field(default=-1)

    def feed_hit(self, hit: HitEvent) -> Side | None:
        if self.phase == "idle":
            self.phase = "await_bounce"
            self.striker = hit.side
            self.target_side = _other_side(hit.side)
            self.last_event_frame = hit.frame_idx
            return None

        if self.phase == "await_return" and hit.side == self.returner_side:
            self.phase = "await_bounce"
            self.striker = hit.side
            self.target_side = _other_side(hit.side)
            self.last_event_frame = hit.frame_idx
            return None

        if self.phase == "await_bounce" and hit.side == self.striker:
            # Duplicate hit candidate near the same contact point.
            if hit.frame_idx - self.last_event_frame <= HIT_DEDUP_FRAMES:
                self.last_event_frame = hit.frame_idx
                return None

        winner = _other_side(hit.side)
        self._reset()
        return winner

    def feed_bounce(self, bounce: BounceEvent) -> Side | None:
        if self.phase == "idle":
            return None

        if self.phase == "await_bounce":
            if bounce.side == self.target_side:
                self.phase = "await_return"
                self.returner_side = bounce.side
                self.striker_side = self.striker
                self.last_event_frame = bounce.frame_idx
                return None

            winner = _other_side(self.striker or bounce.side)
            self._reset()
            return winner

        if self.phase == "await_return":
            if bounce.side == self.returner_side:
                winner = self.striker_side
                self._reset()
                return winner

            winner = self.returner_side
            self._reset()
            return winner

        return None

    def feed_net(self, net: NetEvent) -> Side | None:
        winner = _other_side(net.side)
        self._reset()
        return winner

    def feed_miss(self, miss: MissEvent) -> Side | None:
        winner = _other_side(miss.side)
        self._reset()
        return winner

    def feed_ball_lost(self) -> Side | None:
        if self.phase == "idle":
            return None

        if self.phase == "await_bounce":
            winner = _other_side(self.striker or "left")
        else:
            winner = self.striker_side or "left"

        self._reset()
        return winner

    def feed_timeout(self, frame_idx: int) -> Side | None:
        if self.phase == "idle":
            return None

        if self.phase == "await_bounce":
            if frame_idx - self.last_event_frame < MAX_HIT_TO_BOUNCE_FRAMES:
                return None
            winner = _other_side(self.striker or "left")
            self._reset()
            return winner

        if self.phase == "await_return":
            if frame_idx - self.last_event_frame < MAX_BOUNCE_TO_HIT_FRAMES:
                return None
            winner = self.striker_side or "left"
            self._reset()
            return winner

        return None

    def _reset(self) -> None:
        self.phase = "idle"
        self.striker = None
        self.target_side = None
        self.returner_side = None
        self.striker_side = None
        self.last_event_frame = -1


def _other_side(side: Side) -> Side:
    return "right" if side == "left" else "left"


def _load_tracknet(weights: Path, device: torch.device) -> TrackNet:
    model = TrackNet()
    model.load_state_dict(torch.load(str(weights), map_location=device, weights_only=True))
    return model.to(device).eval()


def _predict_heatmap(
    model: TrackNet,
    frames: list[NDArray[np.float32]],
    device: torch.device,
) -> tuple[float, float, NDArray[np.float32]]:
    stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmap = torch.sigmoid(model(tensor))[0, 0]
    hm: NDArray[np.float32] = heatmap.cpu().numpy()
    if hm.max() < CONF_THRESHOLD:
        return -1.0, -1.0, hm
    flat = int(hm.argmax())
    cy, cx = divmod(flat, TARGET_W)
    return float(cx), float(cy), hm


def _interpolate(
    detections: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    if len(detections) < 2:
        return dict(detections)
    keys = sorted(detections)
    xs = np.array([detections[k][0] for k in keys], dtype=float)
    ys = np.array([detections[k][1] for k in keys], dtype=float)
    k = min(3, len(keys) - 1)
    spl_x = make_interp_spline(keys, xs, k=k)
    spl_y = make_interp_spline(keys, ys, k=k)
    return {i: (float(spl_x(i)), float(spl_y(i))) for i in range(keys[0], keys[-1] + 1)}


def _run_calibration(video_path: Path) -> CalibrationData:
    """Single-pass calibration using unified seg model (table + person).

    Net mask and geometry are derived from the table mask centerline.
    Falls back to defaults if model weights are missing.
    """
    if not SEG_WEIGHTS.exists():
        print(f"  [calibration] {SEG_WEIGHTS} not found — using default geometry")
        default_geo = NetGeometry()
        return CalibrationData(
            net_mask=np.zeros((TARGET_H, TARGET_W), dtype=np.uint8),
            table_mask=np.ones((TARGET_H, TARGET_W), dtype=np.uint8),
            net_geometry=default_geo,
        )

    try:
        print(f"Calibration — {SEG_WEIGHTS.name} …")
        model = load_net_model(SEG_WEIGHTS)
        seg_maps = compute_segmentation_maps(
            video_path=video_path,
            net_model=model,
            n_frames=SEG_FRAMES,
            target_w=TARGET_W,
            target_h=TARGET_H,
        )

        if seg_maps.table_mask is None:
            raise RuntimeError("calibration failed: table not found")

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * TABLE_MASK_MARGIN_PX + 1, 2 * TABLE_MASK_MARGIN_PX + 1),
        )
        table_mask = cv2.dilate(seg_maps.table_mask, kernel, iterations=1)

        net_geometry = seg_maps.net_geometry

        net_mask = _build_net_mask_from_table(table_mask)
        if net_mask is None:
            raise RuntimeError("calibration failed: net not found")
        net_geometry = _net_geometry_from_mask(net_mask)
        print("  [calibration] Net derived from table geometry")

        return CalibrationData(
            net_mask=net_mask,
            table_mask=table_mask,
            net_geometry=net_geometry,
        )
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("calibration failed:"):
            raise
        raise RuntimeError(f"calibration failed: {exc}") from exc


def _net_geometry_from_mask(net_mask: np.ndarray) -> NetGeometry:
    ys, xs = np.where(net_mask > 0)
    if len(xs) == 0:
        raise RuntimeError("calibration failed: net not found")
    return NetGeometry(
        cx=float(xs.mean()) / TARGET_W,
        top_y=float(ys.min()) / TARGET_H,
    )


def _try_update_calibration(
    model: "YOLO",
    frame_bgr: np.ndarray,
    conf: float = 0.25,
    mask_threshold: float = 0.35,
) -> CalibrationData | None:
    """Run YOLO-seg on a single pre-resized (TARGET_H, TARGET_W) frame.

    Returns updated CalibrationData, or None if table is not detected
    (caller should keep the previous calibration in that case).
    """
    from eventnet.segmentation import _label_from_result

    result = model(frame_bgr, conf=conf, verbose=False)[0]
    table_acc = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)
    table_hits = 0

    masks = getattr(result, "masks", None)
    if masks is not None:
        for det_idx, mask_tensor in enumerate(masks.data):
            mask_np = mask_tensor.cpu().numpy()
            m = cv2.resize(mask_np, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            label = _label_from_result(result, det_idx)
            if label in {"0", "table"}:
                table_acc += (m > 0.5).astype(np.float32)
                table_hits += 1

    if table_hits == 0:
        return None

    table_raw = (table_acc / table_hits >= mask_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * TABLE_MASK_MARGIN_PX + 1, 2 * TABLE_MASK_MARGIN_PX + 1),
    )
    table_mask = cv2.dilate(table_raw, kernel, iterations=1)

    net_mask = _build_net_mask_from_table(table_mask)
    if net_mask is None:
        return None
    net_geometry = _net_geometry_from_mask(net_mask)

    return CalibrationData(
        net_mask=net_mask,
        table_mask=table_mask,
        net_geometry=net_geometry,
    )


def _build_net_mask_from_table(table_mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(table_mask > 0)
    if len(xs) == 0:
        return None

    y_min = int(ys.min())
    y_max = int(ys.max())
    center_points: list[tuple[int, int]] = []

    for y in range(y_min, y_max + 1):
        row = np.where(table_mask[y] > 0)[0]
        if len(row) < 2:
            continue
        x_left = int(row.min())
        x_right = int(row.max())
        x_mid = (x_left + x_right) // 2
        center_points.append((x_mid, y))

    if len(center_points) < 8:
        return None

    line_mask = np.zeros_like(table_mask, dtype=np.uint8)
    for (x0, y0), (x1, y1) in zip(center_points, center_points[1:]):
        cv2.line(line_mask, (x0, y0), (x1, y1), 1, 1, cv2.LINE_AA)

    table_widths = []
    for _, y in center_points[:: max(1, len(center_points) // 20)]:
        row = np.where(table_mask[y] > 0)[0]
        if len(row) >= 2:
            table_widths.append(int(row.max()) - int(row.min()))

    thickness = 6
    if table_widths:
        thickness = max(4, int(np.median(table_widths) * 0.035))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
    net_mask = cv2.dilate(line_mask, kernel, iterations=1)
    net_mask = ((net_mask > 0) & (table_mask > 0)).astype(np.uint8)
    return net_mask if int(net_mask.sum()) > 0 else None


def _smooth_signal(values: NDArray[np.float32]) -> NDArray[np.float32]:
    n = len(values)
    if n < 5:
        return values.copy()

    window = min(TRAJ_SMOOTH_WINDOW, n if n % 2 == 1 else n - 1)
    if window < 5:
        return values.copy()

    poly = min(TRAJ_SMOOTH_POLY, window - 1)
    smoothed = savgol_filter(values, window_length=window, polyorder=poly, mode="interp")
    return smoothed.astype(np.float32)


def build_trajectory_series(
    all_positions: dict[int, tuple[float, float]],
) -> TrajectorySeries | None:
    frames = sorted(all_positions)
    if len(frames) < (2 * LOCAL_WINDOW + 3):
        return None

    x = np.array([all_positions[f][0] for f in frames], dtype=np.float32)
    y = np.array([all_positions[f][1] for f in frames], dtype=np.float32)
    x_s = _smooth_signal(x)
    y_s = _smooth_signal(y)
    dx = np.gradient(x_s).astype(np.float32)
    dy = np.gradient(y_s).astype(np.float32)
    ddx = np.gradient(dx).astype(np.float32)
    ddy = np.gradient(dy).astype(np.float32)

    return TrajectorySeries(
        frames=frames,
        x=x,
        y=y,
        x_s=x_s,
        y_s=y_s,
        dx=dx,
        dy=dy,
        ddx=ddx,
        ddy=ddy,
    )


def detect_hits(
    traj: TrajectorySeries,
    net_cx_px: float,
) -> list[HitEvent]:
    hits: list[HitEvent] = []
    net_margin = HIT_NET_MARGIN_RATIO * TARGET_W
    last_frame = -HIT_DEDUP_FRAMES * 2

    for i in range(LOCAL_WINDOW, len(traj.frames) - LOCAL_WINDOW):
        left_dx = float(np.median(traj.dx[i - LOCAL_WINDOW:i]))
        right_dx = float(np.median(traj.dx[i + 1:i + 1 + LOCAL_WINDOW]))
        travel_before = float(abs(traj.x_s[i] - traj.x_s[i - LOCAL_WINDOW]))
        travel_after = float(abs(traj.x_s[i + LOCAL_WINDOW] - traj.x_s[i]))
        cy = float(traj.y_s[i])
        cx = float(traj.x_s[i])

        side: Side | None = None
        if left_dx < -HIT_MIN_X_SPEED and right_dx > HIT_MIN_X_SPEED:
            side = "left"
            if cx >= net_cx_px - net_margin:
                continue
        elif left_dx > HIT_MIN_X_SPEED and right_dx < -HIT_MIN_X_SPEED:
            side = "right"
            if cx <= net_cx_px + net_margin:
                continue
        else:
            continue

        if min(travel_before, travel_after) < HIT_MIN_TRAVEL_PX:
            continue
        if cy > HIT_Y_MAX_RATIO * TARGET_H:
            continue

        frame_idx = traj.frames[i]
        if frame_idx - last_frame < HIT_DEDUP_FRAMES:
            if hits and (travel_before + travel_after) > hits[-1].confidence:
                hits[-1] = HitEvent(
                    frame_idx=frame_idx,
                    side=side,
                    cx=float(traj.x[i]),
                    cy=float(traj.y[i]),
                    confidence=float(travel_before + travel_after),
                )
                last_frame = frame_idx
            continue

        hits.append(HitEvent(
            frame_idx=frame_idx,
            side=side,
            cx=float(traj.x[i]),
            cy=float(traj.y[i]),
            confidence=float(travel_before + travel_after),
        ))
        last_frame = frame_idx

    return hits


def detect_bounces(
    traj: TrajectorySeries,
    net_cx_px: float,
    table_mask: np.ndarray | None,
) -> list[BounceEvent]:
    bounces: list[BounceEvent] = []
    last_frame = -BOUNCE_DEDUP_FRAMES * 2

    for i in range(LOCAL_WINDOW, len(traj.frames) - LOCAL_WINDOW):
        left_dy = float(np.median(traj.dy[i - LOCAL_WINDOW:i]))
        right_dy = float(np.median(traj.dy[i + 1:i + 1 + LOCAL_WINDOW]))
        curvature = float(left_dy - right_dy)
        x_speed = float(abs(traj.dx[i]))
        cy = float(traj.y_s[i])
        cx = float(traj.x_s[i])

        if left_dy <= BOUNCE_MIN_DY or right_dy >= -BOUNCE_MIN_DY:
            continue
        if curvature < BOUNCE_MIN_CURVATURE:
            continue
        if x_speed < BOUNCE_MIN_X_SPEED:
            continue
        if cy < BOUNCE_MIN_Y_RATIO * TARGET_H:
            continue
        if cy > BOUNCE_MAX_Y_RATIO * TARGET_H:
            continue
        if cx <= 0.05 * TARGET_W or cx >= 0.95 * TARGET_W:
            continue
        if table_mask is not None:
            x_i = int(np.clip(round(cx), 0, TARGET_W - 1))
            y_i = int(np.clip(round(cy), 0, TARGET_H - 1))
            if table_mask[y_i, x_i] == 0:
                continue

        frame_idx = traj.frames[i]
        if frame_idx - last_frame < BOUNCE_DEDUP_FRAMES:
            if bounces and curvature > bounces[-1].confidence:
                side: Side = "left" if cx < net_cx_px else "right"
                bounces[-1] = BounceEvent(
                    frame_idx=frame_idx,
                    side=side,
                    cx=float(traj.x[i]),
                    cy=float(traj.y[i]),
                    confidence=curvature,
                )
                last_frame = frame_idx
            continue

        side = "left" if cx < net_cx_px else "right"
        bounces.append(BounceEvent(
            frame_idx=frame_idx,
            side=side,
            cx=float(traj.x[i]),
            cy=float(traj.y[i]),
            confidence=curvature,
        ))
        last_frame = frame_idx

    return bounces


def _filter_hits_near_bounces(
    hits: list[HitEvent],
    bounces: list[BounceEvent],
    radius_frames: int = 6,
) -> list[HitEvent]:
    bounce_frames = [b.frame_idx for b in bounces]
    filtered: list[HitEvent] = []
    for hit in hits:
        if any(abs(hit.frame_idx - bf) <= radius_frames for bf in bounce_frames):
            continue
        filtered.append(hit)
    return filtered


def detect_net_events(
    traj: TrajectorySeries,
    hits: list[HitEvent],
    bounces: list[BounceEvent],
    net_cx_px: float,
    net_mask: np.ndarray | None,
) -> list[NetEvent]:
    nets: list[NetEvent] = []
    if not hits:
        return nets

    net_near_px = NET_NEAR_RATIO * TARGET_W
    cross_margin = NET_MAX_CROSS_RATIO * TARGET_W
    bounces_by_frame = {b.frame_idx: b for b in bounces}
    last_frame = -NET_DEDUP_FRAMES * 2

    for idx, hit in enumerate(hits):
        next_hit_frame = hits[idx + 1].frame_idx if idx + 1 < len(hits) else traj.frames[-1] + 1
        segment_bounces = [
            b for b in bounces
            if hit.frame_idx < b.frame_idx < next_hit_frame
        ]
        if any(b.side == _other_side(hit.side) for b in segment_bounces):
            continue

        start = np.searchsorted(traj.frames, hit.frame_idx)
        stop = np.searchsorted(traj.frames, next_hit_frame)
        if stop - start < 3:
            continue

        seg_x = traj.x_s[start:stop]
        seg_y = traj.y_s[start:stop]
        seg_dx = traj.dx[start:stop]
        seg_frames = traj.frames[start:stop]
        seg_limit = min(len(seg_frames), NET_POST_HIT_FRAMES)
        if seg_limit < 3:
            continue
        seg_x = seg_x[:seg_limit]
        seg_y = seg_y[:seg_limit]
        seg_dx = seg_dx[:seg_limit]
        seg_frames = seg_frames[:seg_limit]

        near_idx = int(np.argmin(np.abs(seg_x - net_cx_px)))
        near_frame = seg_frames[near_idx]
        near_x = float(seg_x[near_idx])
        near_y = float(seg_y[near_idx])
        near_dist = abs(near_x - net_cx_px)
        hit_to_net_progress = abs(net_cx_px - hit.cx)
        progress = abs(near_x - hit.cx)
        near_on_net = False
        if net_mask is not None:
            x_i = int(np.clip(round(near_x), 0, TARGET_W - 1))
            y_i = int(np.clip(round(near_y), 0, TARGET_H - 1))
            near_on_net = bool(net_mask[y_i, x_i] > 0)

        crossed = False
        if hit.side == "left":
            crossed = bool(np.max(seg_x) >= net_cx_px + cross_margin)
        else:
            crossed = bool(np.min(seg_x) <= net_cx_px - cross_margin)

        reversal_near_net = False
        if 1 <= near_idx < len(seg_dx) - 1:
            before = float(np.median(seg_dx[max(0, near_idx - 2):near_idx + 1]))
            after = float(np.median(seg_dx[near_idx:min(len(seg_dx), near_idx + 3)]))
            reversal_near_net = (
                abs(before) >= NET_REVERSAL_X_SPEED
                and abs(after) >= NET_REVERSAL_X_SPEED
                and np.sign(before) != np.sign(after)
            )

        same_side_bounce_near_net = any(
            b.side == hit.side and abs(b.cx - net_cx_px) <= 1.5 * net_near_px
            for b in segment_bounces
        )

        y_ok = NET_MIN_Y_RATIO * TARGET_H <= near_y <= NET_MAX_Y_RATIO * TARGET_H
        progress_ok = progress >= NET_MIN_PROGRESS_RATIO * max(hit_to_net_progress, 1.0)
        stagnated_before_cross = not crossed and progress_ok
        candidate_near_net = (near_dist <= net_near_px and y_ok) or near_on_net

        if crossed and not reversal_near_net and not same_side_bounce_near_net:
            continue
        if not candidate_near_net and not same_side_bounce_near_net:
            continue
        if not reversal_near_net and not same_side_bounce_near_net and not stagnated_before_cross:
            continue
        if any(abs(near_frame - b.frame_idx) <= 3 for b in segment_bounces):
            continue

        if near_frame - last_frame < NET_DEDUP_FRAMES:
            continue

        # If there is a bounce at the same frame, prefer bounce over net.
        if near_frame in bounces_by_frame:
            continue

        nets.append(NetEvent(
            frame_idx=near_frame,
            side=hit.side,
            cx=float(traj.x[start + near_idx]),
            cy=float(traj.y[start + near_idx]),
            confidence=float(max(1.0, net_near_px - near_dist)),
        ))
        last_frame = near_frame

    return nets


def detect_miss_events(
    traj: TrajectorySeries,
    hits: list[HitEvent],
    bounces: list[BounceEvent],
    net_cx_px: float,
) -> list[MissEvent]:
    misses: list[MissEvent] = []
    if not hits:
        return misses

    cross_margin = NET_MAX_CROSS_RATIO * TARGET_W
    out_margin = MISS_OUT_X_MARGIN_RATIO * TARGET_W

    for idx, hit in enumerate(hits):
        target_side = _other_side(hit.side)
        next_hit_frame = hits[idx + 1].frame_idx if idx + 1 < len(hits) else traj.frames[-1] + 1

        segment_bounces = [
            b for b in bounces
            if hit.frame_idx < b.frame_idx < next_hit_frame
        ]
        if any(b.side == target_side for b in segment_bounces):
            continue

        start = np.searchsorted(traj.frames, hit.frame_idx)
        stop = np.searchsorted(traj.frames, next_hit_frame)
        if stop - start < 3:
            continue

        seg_frames = traj.frames[start:stop]
        seg_x = traj.x_s[start:stop]
        seg_y = traj.y_s[start:stop]
        seg_dx = traj.dx[start:stop]

        seg_limit = min(len(seg_frames), MAX_HIT_TO_BOUNCE_FRAMES)
        if seg_limit < 3:
            continue
        seg_frames = seg_frames[:seg_limit]
        seg_x = seg_x[:seg_limit]
        seg_y = seg_y[:seg_limit]
        seg_dx = seg_dx[:seg_limit]

        if hit.side == "left":
            crossed_mask = seg_x >= net_cx_px + cross_margin
            returned_mask = seg_dx < -MISS_RETURN_X_SPEED
        else:
            crossed_mask = seg_x <= net_cx_px - cross_margin
            returned_mask = seg_dx > MISS_RETURN_X_SPEED

        if not bool(np.any(crossed_mask)):
            continue

        crossed_idx = int(np.argmax(crossed_mask))
        post_frames = seg_frames[crossed_idx:]
        post_x = seg_x[crossed_idx:]
        post_y = seg_y[crossed_idx:]
        post_returned = returned_mask[crossed_idx:]

        low_mask = post_y >= MISS_MAX_Y_RATIO * TARGET_H
        out_mask = (post_x <= out_margin) | (post_x >= TARGET_W - out_margin)

        terminal_idx: int | None = None
        if bool(np.any(low_mask)):
            terminal_idx = crossed_idx + int(np.argmax(low_mask))
        elif bool(np.any(out_mask)):
            terminal_idx = crossed_idx + int(np.argmax(out_mask))
        elif bool(np.any(post_returned)):
            terminal_idx = crossed_idx + int(np.argmax(post_returned))
        elif len(seg_frames) == seg_limit:
            terminal_idx = len(seg_frames) - 1

        if terminal_idx is None or terminal_idx <= crossed_idx:
            continue

        terminal_frame = seg_frames[terminal_idx]
        if any(abs(terminal_frame - b.frame_idx) <= 4 for b in segment_bounces):
            continue

        misses.append(MissEvent(
            frame_idx=terminal_frame,
            side=hit.side,
            cx=float(traj.x[start + terminal_idx]),
            cy=float(traj.y[start + terminal_idx]),
            confidence=float(terminal_idx - crossed_idx),
        ))

    deduped: list[MissEvent] = []
    for miss in misses:
        if deduped and miss.frame_idx - deduped[-1].frame_idx <= HIT_DEDUP_FRAMES:
            if miss.confidence > deduped[-1].confidence:
                deduped[-1] = miss
            continue
        deduped.append(miss)
    return deduped


def merge_events(
    hits: list[HitEvent],
    bounces: list[BounceEvent],
    nets: list[NetEvent],
    misses: list[MissEvent],
) -> list[TrajectoryEvent]:
    events = [
        TrajectoryEvent(
            frame_idx=h.frame_idx,
            kind="hit",
            side=h.side,
            cx=h.cx,
            cy=h.cy,
            confidence=h.confidence,
        )
        for h in hits
    ]
    events.extend(
        TrajectoryEvent(
            frame_idx=b.frame_idx,
            kind="bounce",
            side=b.side,
            cx=b.cx,
            cy=b.cy,
            confidence=b.confidence,
        )
        for b in bounces
    )
    events.extend(
        TrajectoryEvent(
            frame_idx=n.frame_idx,
            kind="net",
            side=n.side,
            cx=n.cx,
            cy=n.cy,
            confidence=n.confidence,
        )
        for n in nets
    )
    events.extend(
        TrajectoryEvent(
            frame_idx=m.frame_idx,
            kind="miss",
            side=m.side,
            cx=m.cx,
            cy=m.cy,
            confidence=m.confidence,
        )
        for m in misses
    )

    priority = {"net": 0, "miss": 1, "bounce": 2, "hit": 3}
    events.sort(key=lambda e: (e.frame_idx, priority[e.kind]))

    deduped: list[TrajectoryEvent] = []
    for event in events:
        if deduped and abs(event.frame_idx - deduped[-1].frame_idx) <= 1 and event.kind == deduped[-1].kind:
            if event.confidence > deduped[-1].confidence:
                deduped[-1] = event
            continue
        deduped.append(event)
    return deduped


def build_score_timeline(
    events: list[TrajectoryEvent],
    detections: dict[int, tuple[float, float]],
    total_frames: int,
) -> dict[int, FrameScore]:
    machine = RallyStateMachine()
    score = FrameScore()
    timeline: dict[int, FrameScore] = {}

    event_map: dict[int, list[TrajectoryEvent]] = {}
    for event in events:
        event_map.setdefault(event.frame_idx, []).append(event)

    consecutive_missing = 0
    last_label: str | None = None
    last_label_frame = -EVENT_TEXT_FADE_FRAMES * 2

    for fi in range(total_frames):
        if fi in detections:
            consecutive_missing = 0
        else:
            consecutive_missing += 1

        winner: Side | None = machine.feed_timeout(fi)
        if winner is not None:
            score = FrameScore(
                left=score.left + (1 if winner == "left" else 0),
                right=score.right + (1 if winner == "right" else 0),
                point_winner=winner,
                event_label=f"point {winner}",
            )
            last_label = score.event_label
            last_label_frame = fi

        for event in event_map.get(fi, []) if winner is None else []:
            last_label = f"{event.kind} {event.side}"
            last_label_frame = fi

            if event.kind == "hit":
                winner = machine.feed_hit(HitEvent(**event.model_dump()))
            elif event.kind == "bounce":
                winner = machine.feed_bounce(BounceEvent(**event.model_dump()))
            elif event.kind == "miss":
                winner = machine.feed_miss(MissEvent(**event.model_dump()))
            else:
                winner = machine.feed_net(NetEvent(**event.model_dump()))

            if winner is not None:
                score = FrameScore(
                    left=score.left + (1 if winner == "left" else 0),
                    right=score.right + (1 if winner == "right" else 0),
                    point_winner=winner,
                    event_label=last_label,
                )
                last_label = f"point {winner}"
                last_label_frame = fi
                break

        if winner is None and consecutive_missing >= BALL_LOST_FRAMES:
            winner = machine.feed_ball_lost()
            consecutive_missing = 0
            if winner is not None:
                score = FrameScore(
                    left=score.left + (1 if winner == "left" else 0),
                    right=score.right + (1 if winner == "right" else 0),
                    point_winner=winner,
                    event_label=f"point {winner}",
                )
                last_label = score.event_label
                last_label_frame = fi

        label = last_label if fi - last_label_frame <= EVENT_TEXT_FADE_FRAMES else None
        timeline[fi] = FrameScore(
            left=score.left,
            right=score.right,
            point_winner=score.point_winner if fi == last_label_frame and winner is not None else None,
            event_label=label,
        )

        if score.point_winner is not None and fi != last_label_frame:
            score = FrameScore(left=score.left, right=score.right)

    return timeline


_COLOR_LEFT: tuple[int, int, int] = (255, 160, 40)
_COLOR_RIGHT: tuple[int, int, int] = (40, 200, 255)
_COLOR_POINT: tuple[int, int, int] = (50, 220, 80)
_COLOR_SCORE_BG: tuple[int, int, int] = (25, 25, 25)
_COLOR_HIT: tuple[int, int, int] = (80, 255, 180)
_COLOR_BOUNCE: tuple[int, int, int] = (255, 220, 0)
_COLOR_NET: tuple[int, int, int] = (0, 80, 230)
_COLOR_MISS: tuple[int, int, int] = (40, 80, 255)
_RUSSIAN_VOICE = "Milena"


def _draw_score_bar(frame: np.ndarray, fs: FrameScore) -> None:
    bar_h = 38
    bg = _COLOR_POINT if fs.point_winner else _COLOR_SCORE_BG
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (TARGET_W, bar_h), bg, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    score_text = f"{fs.left}  :  {fs.right}"
    (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 0.90, 2)
    tx = (TARGET_W - tw) // 2
    cv2.putText(frame, score_text, (tx, bar_h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.90, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "LEFT", (8, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _COLOR_LEFT, 1, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT", (TARGET_W - 56, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _COLOR_RIGHT, 1, cv2.LINE_AA)

    if fs.event_label:
        cv2.putText(frame, fs.event_label.upper(), (10, TARGET_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, fs.event_label.upper(), (10, TARGET_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_bounce_marker(frame: np.ndarray, b: BounceEvent, current_frame: int, fade_frames: int = 40) -> None:
    age = current_frame - b.frame_idx
    if age < 0 or age > fade_frames:
        return
    alpha = 1.0 - age / fade_frames
    overlay = frame.copy()
    cv2.circle(overlay, (int(b.cx), int(b.cy)), max(4, int(10 * alpha)), _COLOR_BOUNCE, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)


def _draw_hit_marker(frame: np.ndarray, h: HitEvent, current_frame: int, fade_frames: int = 30) -> None:
    age = current_frame - h.frame_idx
    if age < 0 or age > fade_frames:
        return
    alpha = 1.0 - age / fade_frames
    radius = max(5, int(12 * alpha))
    overlay = frame.copy()
    cv2.circle(overlay, (int(h.cx), int(h.cy)), radius, _COLOR_HIT, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)


def _draw_net_marker(frame: np.ndarray, n: NetEvent, current_frame: int, fade_frames: int = 36) -> None:
    age = current_frame - n.frame_idx
    if age < 0 or age > fade_frames:
        return
    alpha = 1.0 - age / fade_frames
    overlay = frame.copy()
    pt = (int(n.cx), int(n.cy))
    size = max(7, int(14 * alpha))
    cv2.line(overlay, (pt[0] - size, pt[1] - size), (pt[0] + size, pt[1] + size), _COLOR_NET, 2, cv2.LINE_AA)
    cv2.line(overlay, (pt[0] - size, pt[1] + size), (pt[0] + size, pt[1] - size), _COLOR_NET, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha * 0.90, frame, 1 - alpha * 0.90, 0, frame)


def _draw_miss_marker(frame: np.ndarray, m: MissEvent, current_frame: int, fade_frames: int = 36) -> None:
    age = current_frame - m.frame_idx
    if age < 0 or age > fade_frames:
        return
    alpha = 1.0 - age / fade_frames
    overlay = frame.copy()
    pt = (int(m.cx), int(m.cy))
    radius = max(7, int(14 * alpha))
    cv2.circle(overlay, pt, radius, _COLOR_MISS, 2, cv2.LINE_AA)
    cv2.putText(overlay, "OUT", (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _COLOR_MISS, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha * 0.90, frame, 1 - alpha * 0.90, 0, frame)


def _draw_trail(frame: np.ndarray, trail: list[tuple[float, float]]) -> None:
    if len(trail) < 2:
        return
    if len(trail) >= 4:
        idx_arr = np.arange(len(trail), dtype=float)
        k = min(3, len(trail) - 1)
        spl_x = make_interp_spline(idx_arr, [p[0] for p in trail], k=k)
        spl_y = make_interp_spline(idx_arr, [p[1] for p in trail], k=k)
        t = np.linspace(0, len(trail) - 1, 60)
        curve = [(int(spl_x(v)), int(spl_y(v))) for v in t]
    else:
        curve = [(int(p[0]), int(p[1])) for p in trail]
    for j in range(1, len(curve)):
        alpha = j / len(curve)
        cv2.line(frame, curve[j - 1], curve[j], (int(255 * alpha), int(200 * alpha), 0), max(1, int(alpha * 3)), cv2.LINE_AA)


def _draw_ball(frame: np.ndarray, cx: float, cy: float, detected: bool) -> None:
    color = (0, 220, 255) if detected else (160, 160, 220)
    radius = 7 if detected else 5
    cv2.circle(frame, (int(cx), int(cy)), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(cx), int(cy)), radius, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_net_mask(frame: np.ndarray, net_mask: np.ndarray | None) -> None:
    if net_mask is None:
        return
    contours, _ = cv2.findContours(net_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame.copy()
    overlay[net_mask > 0] = (180, 180, 255)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.drawContours(frame, contours, -1, (180, 180, 255), 1, cv2.LINE_AA)


def _draw_table_outline(
    frame: np.ndarray,
    table_mask: np.ndarray,
) -> None:
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame.copy()
    overlay[table_mask > 0] = (0, 128, 255)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.drawContours(frame, contours, -1, (0, 180, 255), 1, cv2.LINE_AA)


def _to_heatmap(hm: NDArray[np.float32]) -> np.ndarray:
    return cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)  # type: ignore[return-value]


def _extract_point_events(timeline: dict[int, FrameScore]) -> list[tuple[int, Side]]:
    return [
        (frame_idx, frame_score.point_winner)
        for frame_idx, frame_score in sorted(timeline.items())
        if frame_score.point_winner is not None
    ]


def _read_wav_pcm(path: Path) -> tuple[int, NDArray[np.float32]]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    return sample_rate, audio


def _write_wav_pcm(path: Path, sample_rate: int, audio: NDArray[np.float32]) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _synthesize_score_audio(
    point_events: list[tuple[int, Side]],
    fps: float,
    total_frames: int,
    out_wav: Path,
) -> bool:
    if not point_events or fps <= 0:
        return False

    phrases: dict[Side, str] = {
        "left": "очко левому игроку",
        "right": "очко правому игроку",
    }

    with tempfile.TemporaryDirectory(prefix="score_tts_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        wav_paths: dict[Side, Path] = {
            "left": tmp_dir / "left.wav",
            "right": tmp_dir / "right.wav",
        }

        phrase_audio: dict[Side, NDArray[np.float32]] = {}
        sample_rate = 22_050

        for side, text in phrases.items():
            aiff_path = tmp_dir / f"{side}.aiff"
            subprocess.run(
                [
                    "say",
                    "-v",
                    _RUSSIAN_VOICE,
                    "-o",
                    str(aiff_path),
                    text,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(aiff_path),
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    str(wav_paths[side]),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            sample_rate, phrase_audio[side] = _read_wav_pcm(wav_paths[side])

        total_samples = int((total_frames / fps) * sample_rate) + sample_rate
        mixed = np.zeros(total_samples, dtype=np.float32)

        for frame_idx, winner in point_events:
            clip = phrase_audio[winner]
            start = int((frame_idx / fps) * sample_rate)
            end = min(start + len(clip), len(mixed))
            mixed[start:end] += clip[: end - start]

        _write_wav_pcm(out_wav, sample_rate, mixed)
        return True


def _attach_score_audio(
    silent_video: Path,
    output_video: Path,
    timeline: dict[int, FrameScore],
    fps: float,
    total_frames: int,
) -> None:
    point_events = _extract_point_events(timeline)
    if not point_events:
        silent_video.replace(output_video)
        return

    with tempfile.TemporaryDirectory(prefix="score_mux_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        wav_path = tmp_dir / "score_audio.wav"
        created = _synthesize_score_audio(point_events, fps, total_frames, wav_path)
        if not created:
            silent_video.replace(output_video)
            return

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(silent_video),
                "-i",
                str(wav_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    silent_video.unlink(missing_ok=True)


def _run_eventnet(
    hm_cache: dict[int, NDArray[np.float32]],
    all_positions: dict[int, tuple[float, float]],
    net_cx_px: float,
    weights: Path,
    device: torch.device,
    threshold: float = EVENTNET_THRESHOLD,
    window_size: int = 15,
    dedup_frames: int = EVENTNET_DEDUP,
) -> tuple[list[HitEvent], list[BounceEvent]]:
    """Slide EventNet over cached TrackNet heatmaps → hit + bounce event lists.

    Heatmaps are downscaled from (360, 640) to EventNet's (36, 64).
    Missing frames (between INFER_STEP detections) are zeroed — consistent
    with the 50 % heatmap dropout used during training.
    """
    from eventnet.heatmap import HM_H, HM_W
    from eventnet.model import HeatmapEventNet, HeatmapEventNetConfig

    checkpoint = torch.load(str(weights), map_location=device, weights_only=True)
    cfg = HeatmapEventNetConfig(**checkpoint["cfg"])
    model = HeatmapEventNet(cfg).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])

    frames = sorted(hm_cache.keys())
    half = window_size // 2
    empty = np.zeros((HM_H, HM_W), dtype=np.float32)

    hm_small: dict[int, NDArray[np.float32]] = {
        fi: cv2.resize(hm, (HM_W, HM_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        for fi, hm in hm_cache.items()
    }

    bounce_scores: dict[int, float] = {}

    with torch.no_grad():
        for i in range(half, len(frames) - half):
            fi = frames[i]
            window_frames = frames[i - half: i + half + 1]
            window = np.stack(
                [hm_small.get(f, empty) for f in window_frames],
                axis=0,
            )  # (window_size, HM_H, HM_W)
            tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            probs = torch.sigmoid(model(tensor)[0]).cpu().numpy()
            bounce_scores[fi] = float(probs[0])  # bounce = idx 0

    def _peak_pick(scores: dict[int, float]) -> list[tuple[int, float]]:
        peaks: list[tuple[int, float]] = []
        for fi, score in sorted(scores.items()):
            if score < threshold:
                continue
            if peaks and fi - peaks[-1][0] < dedup_frames:
                if score > peaks[-1][1]:
                    peaks[-1] = (fi, score)
            else:
                peaks.append((fi, score))
        return peaks

    bounces: list[BounceEvent] = []
    for fi, conf in _peak_pick(bounce_scores):
        cx, cy = all_positions.get(fi, (-1.0, -1.0))
        if cx < 0:
            continue
        side: Side = "left" if cx < net_cx_px else "right"
        bounces.append(BounceEvent(frame_idx=fi, side=side, cx=cx, cy=cy, confidence=conf))

    return [], bounces


def run(video_path: Path, output: Path) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    tracknet = _load_tracknet(TRACKNET_WEIGHTS, device)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    seg_model = load_net_model(SEG_WEIGHTS) if SEG_WEIGHTS.exists() else None
    calibration = _run_calibration(video_path)
    table_mask = calibration.table_mask
    net_mask   = calibration.net_mask
    net_cx_px  = calibration.net_geometry.cx * TARGET_W

    print(f"Pass 1 — TrackNet on {total} frames…")
    detections: dict[int, tuple[float, float]] = {}
    hm_cache: dict[int, NDArray[np.float32]] = {}
    frame_buffer: list[NDArray[np.float32]] = []

    cap = cv2.VideoCapture(str(video_path))
    for fi in tqdm(range(total), desc="TrackNet", unit="frame"):
        ret, raw = cap.read()
        if not ret:
            break

        if fi % SEG_INTERVAL == 0 and fi > 0 and seg_model is not None:
            frame_bgr = cv2.resize(raw, (TARGET_W, TARGET_H))
            updated = _try_update_calibration(seg_model, frame_bgr)
            if updated is not None:
                table_mask = updated.table_mask
                net_mask   = updated.net_mask
                net_cx_px  = updated.net_geometry.cx * TARGET_W

        small = cv2.resize(raw, (TARGET_W, TARGET_H)).astype(np.float32) / 255.0
        frame_buffer.append(small)
        if len(frame_buffer) > 3:
            frame_buffer.pop(0)
        while len(frame_buffer) < 3:
            frame_buffer.insert(0, frame_buffer[0])
        if fi % INFER_STEP == 0:
            cx, cy, hm = _predict_heatmap(tracknet, frame_buffer[-3:], device)
            hm_cache[fi] = hm
            if cx >= 0:
                detections[fi] = (cx, cy)
    cap.release()

    all_positions = _interpolate(detections)
    print(f"  detected {len(detections)} / interpolated {len(all_positions)} frames")

    traj = build_trajectory_series(all_positions)
    if traj is None:
        raise RuntimeError("Not enough tracked points to score the video.")

    print("Pass 2 — event detection…")
    bounces = detect_bounces(traj, net_cx_px, table_mask)

    if EVENTNET_WEIGHTS.exists():
        print(f"  EventNet ({EVENTNET_WEIGHTS.name}) …")
        en_hits, en_bounces = _run_eventnet(hm_cache, all_positions, net_cx_px, EVENTNET_WEIGHTS, device)
        hits = en_hits if en_hits else detect_hits(traj, net_cx_px)
        if en_bounces:
            bounces = en_bounces
        print(f"  EventNet: {len(hits)} hits, {len(bounces)} bounces")
    else:
        hits = detect_hits(traj, net_cx_px)
        print(f"  Trajectory heuristics (no EventNet weights)")

    hits  = _filter_hits_near_bounces(hits, bounces)
    nets  = detect_net_events(traj, hits, bounces, net_cx_px, net_mask)
    misses = detect_miss_events(traj, hits, bounces, net_cx_px)
    events = merge_events(hits, bounces, nets, misses)

    print(f"  hits:    {len(hits)}")
    print(f"  bounces: {len(bounces)}")
    print(f"  nets:    {len(nets)}")
    print(f"  misses:  {len(misses)}")
    for event in events:
        print(
            f"    frame {event.frame_idx:5d}  "
            f"{event.kind:6s}  side={event.side:5s}  "
            f"cx={event.cx:6.1f}  cy={event.cy:6.1f}"
        )

    print("Pass 3 — scoring…")
    timeline = build_score_timeline(events, detections, total)
    final = timeline.get(total - 1, FrameScore())
    print(f"  Final score: LEFT {final.left} : RIGHT {final.right}")

    print("Pass 4 — rendering…")
    silent_output = output.with_name(f"{output.stem}.silent.mp4")
    writer = cv2.VideoWriter(
        str(silent_output),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (TARGET_W * 2, TARGET_H),
    )

    known_frames = sorted(detections)
    heatmap: NDArray[np.float32] = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    for fi in tqdm(range(total), desc="Rendering", unit="frame"):
        ret, raw = cap.read()
        if not ret:
            break

        if fi in hm_cache:
            heatmap = hm_cache[fi]

        left: np.ndarray = cv2.resize(raw, (TARGET_W, TARGET_H))

        trail_keys = [i for i in known_frames if i <= fi][-TRAIL_WINDOW:]
        _draw_trail(left, [detections[i] for i in trail_keys])

        if fi in detections:
            _draw_ball(left, *detections[fi], detected=True)
        elif fi in all_positions:
            _draw_ball(left, *all_positions[fi], detected=False)

        for bounce in bounces:
            _draw_bounce_marker(left, bounce, fi)
        for hit in hits:
            _draw_hit_marker(left, hit, fi)
        for net in nets:
            _draw_net_marker(left, net, fi)
        for miss in misses:
            _draw_miss_marker(left, miss, fi)

        _draw_net_mask(left, net_mask)
        _draw_table_outline(left, table_mask)
        _draw_score_bar(left, timeline.get(fi, FrameScore()))

        right = _to_heatmap(heatmap)
        combined = np.hstack([left, right])
        cv2.line(combined, (TARGET_W, 0), (TARGET_W, TARGET_H - 1), (60, 60, 60), 2)
        writer.write(combined)

    cap.release()
    writer.release()
    _attach_score_audio(silent_output, output, timeline, fps, total)
    print(f"Saved -> {output}")


def main() -> None:
    run(VIDEO_PATH, OUTPUT)


if __name__ == "__main__":
    main()
