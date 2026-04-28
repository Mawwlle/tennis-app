"""Streaming score solver: single-pass detection + EventNet + trajectory scoring.

Pipeline:
  Pass 1  TrackNet (every INFER_STEP) + spline interpolation
          + EventNet causal window -> ball positions + bounce events
          + trajectory heuristics -> hit / net / miss events
          + rally state machine   -> score timeline
  Pass 2  Render annotated MP4 with score overlay + audio

Net geometry is fixed via NET_CX_RATIO — no segmentation required.
Set NET_CX_RATIO to the net's horizontal centre as a fraction of frame width.

Usage:
    uv run python score.py
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from tqdm import tqdm

from tracknet.model import TrackNet

if TYPE_CHECKING:
    from eventnet.model import HeatmapEventNet
    from ultralytics import YOLO

VIDEO_PATH       = Path("test_3.mp4")
TRACKNET_WEIGHTS = Path("weights/tracknet_best.pt")
EVENTNET_WEIGHTS = Path("weights/eventnet_best.pt")
SEG_WEIGHTS      = Path("weights/seg_best.pt")
OUTPUT           = Path("score_result.mp4")

NET_CX_RATIO = 0.50   # net centre as fraction of frame width

TARGET_W = 640
TARGET_H = 360
CONF_THRESHOLD = 0.99
TRAIL_WINDOW   = 18
INFER_STEP     = 4

# EventNet causal window
EN_WINDOW = 15
EN_HALF   = EN_WINDOW // 2
EVENTNET_THRESHOLD = 0.35
EVENTNET_DEDUP     = 8

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


_COLOR_TABLE:  tuple[int, int, int] = (0, 128, 255)   # orange
_COLOR_PERSON: tuple[int, int, int] = (0, 220, 80)    # green


def _load_seg_model(weights: Path) -> "YOLO":
    from ultralytics import YOLO  # type: ignore[reportMissingImports]
    return YOLO(str(weights))


def _predict_seg_masks(
    model: "YOLO",
    frame_bgr: np.ndarray,
    conf: float = 0.25,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (table_mask, person_mask) as uint8 arrays the same size as frame_bgr."""
    h, w = frame_bgr.shape[:2]
    result = model(frame_bgr, conf=conf, verbose=False)[0]
    if result.masks is None:
        return None, None

    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    table_mask = np.zeros((h, w), dtype=np.uint8)
    person_mask = np.zeros((h, w), dtype=np.uint8)

    for det_idx, mask_tensor in enumerate(result.masks.data):
        cls_idx = int(boxes.cls[det_idx].item()) if boxes is not None else -1
        label = str(names.get(cls_idx, cls_idx)).lower() if isinstance(names, dict) else str(cls_idx)

        mask_np = mask_tensor.cpu().numpy()
        resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (resized > 0.5).astype(np.uint8)

        if label in {"0", "table"}:
            table_mask = cv2.bitwise_or(table_mask, binary)
        elif label in {"1", "person"}:
            person_mask = cv2.bitwise_or(person_mask, binary)

    table_out = table_mask if table_mask.any() else None
    person_out = person_mask if person_mask.any() else None
    return table_out, person_out


def _draw_table_mask(frame: np.ndarray, mask: np.ndarray) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame.copy()
    overlay[mask > 0] = _COLOR_TABLE
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.drawContours(frame, contours, -1, _COLOR_TABLE, 1, cv2.LINE_AA)


def _draw_person_mask(frame: np.ndarray, mask: np.ndarray) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame.copy()
    overlay[mask > 0] = _COLOR_PERSON
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.drawContours(frame, contours, -1, _COLOR_PERSON, 1, cv2.LINE_AA)


def _load_tracknet(weights: Path, device: torch.device) -> TrackNet:
    model = TrackNet()
    model.load_state_dict(torch.load(str(weights), map_location=device, weights_only=True))
    return model.to(device).eval()


def _load_eventnet(
    weights: Path,
    device: torch.device,
) -> "HeatmapEventNet":
    from eventnet.model import HeatmapEventNet, HeatmapEventNetConfig

    checkpoint = torch.load(str(weights), map_location=device, weights_only=True)
    cfg = HeatmapEventNetConfig(**checkpoint["cfg"])
    model = HeatmapEventNet(cfg).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model


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


def _run_eventnet_causal(
    model: "HeatmapEventNet",
    hm_cache: dict[int, NDArray[np.float32]],
    all_positions: dict[int, tuple[float, float]],
    net_cx_px: float,
    device: torch.device,
    threshold: float = EVENTNET_THRESHOLD,
    dedup_frames: int = EVENTNET_DEDUP,
) -> list[BounceEvent]:
    """Slide a centered EN_WINDOW over sorted heatmap frames; bounce attributed to center."""
    from eventnet.heatmap import HM_H, HM_W

    frames = sorted(hm_cache.keys())
    if len(frames) < EN_WINDOW:
        return []

    empty = np.zeros((HM_H, HM_W), dtype=np.float32)

    hm_small: dict[int, NDArray[np.float32]] = {
        fi: cv2.resize(hm, (HM_W, HM_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        for fi, hm in hm_cache.items()
    }

    scores: dict[int, float] = {}
    with torch.no_grad():
        for i in range(EN_HALF, len(frames) - EN_HALF):
            fi = frames[i]  # center frame — bounce is attributed here
            window_frames = frames[i - EN_HALF: i + EN_HALF + 1]
            window = np.stack(
                [hm_small.get(f, empty) for f in window_frames],
                axis=0,
            )  # (EN_WINDOW, HM_H, HM_W)
            tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            probs = torch.sigmoid(model(tensor)[0]).cpu().numpy()
            scores[fi] = float(probs[0])  # bounce = idx 0

    bounces: list[BounceEvent] = []
    last_frame = -dedup_frames * 2
    for fi, score in sorted(scores.items()):
        if score < threshold:
            continue
        if fi - last_frame < dedup_frames:
            if bounces and score > bounces[-1].confidence:
                cx, cy = all_positions.get(fi, (-1.0, -1.0))
                bounces[-1] = BounceEvent(
                    frame_idx=fi,
                    side="left" if cx < net_cx_px else "right",
                    cx=cx,
                    cy=cy,
                    confidence=score,
                )
                last_frame = fi
            continue

        cx, cy = all_positions.get(fi, (-1.0, -1.0))
        if cx < 0:
            continue
        bounces.append(BounceEvent(
            frame_idx=fi,
            side="left" if cx < net_cx_px else "right",
            cx=cx,
            cy=cy,
            confidence=score,
        ))
        last_frame = fi

    return bounces


def _smooth_signal(values: NDArray[np.float32]) -> NDArray[np.float32]:
    n = len(values)
    if n < 5:
        return values.copy()
    window = min(TRAJ_SMOOTH_WINDOW, n if n % 2 == 1 else n - 1)
    if window < 5:
        return values.copy()
    poly = min(TRAJ_SMOOTH_POLY, window - 1)
    return savgol_filter(values, window_length=window, polyorder=poly, mode="interp").astype(np.float32)


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


def detect_hits_from_bounces(
    bounces: list[BounceEvent],
    all_positions: dict[int, tuple[float, float]],
    net_cx_px: float,
) -> list[HitEvent]:
    """Infer hits from consecutive bounces on opposite sides.

    Between a bounce on side X (frame f0) and the next bounce on side Y (frame f1, X≠Y),
    player X returned the ball. The hit is at the x-extremum of the trajectory in [f0, f1]:
      - right player: maximum x (ball travels furthest right toward player)
      - left player: minimum x (ball travels furthest left toward player)
    """
    if len(bounces) < 2:
        return []

    sorted_bounces = sorted(bounces, key=lambda b: b.frame_idx)
    hits: list[HitEvent] = []

    for i in range(len(sorted_bounces) - 1):
        b0 = sorted_bounces[i]
        b1 = sorted_bounces[i + 1]

        if b0.side == b1.side:
            continue  # double bounce same side — no return in between

        hitter_side: Side = b0.side  # player on this side returns after the bounce
        f0, f1 = b0.frame_idx, b1.frame_idx

        frames_in = [f for f in all_positions if f0 < f < f1]
        if len(frames_in) < 3:
            continue

        frames_arr = sorted(frames_in)
        xs = np.array([all_positions[f][0] for f in frames_arr], dtype=np.float32)
        ys = np.array([all_positions[f][1] for f in frames_arr], dtype=np.float32)

        if hitter_side == "right":
            hit_idx = int(np.argmax(xs))
            if xs[hit_idx] <= net_cx_px:
                continue  # extremum still on wrong side of net
        else:
            hit_idx = int(np.argmin(xs))
            if xs[hit_idx] >= net_cx_px:
                continue

        hits.append(HitEvent(
            frame_idx=frames_arr[hit_idx],
            side=hitter_side,
            cx=float(xs[hit_idx]),
            cy=float(ys[hit_idx]),
            confidence=float(abs(xs[hit_idx] - net_cx_px)),
        ))

    return hits


def detect_bounces(
    traj: TrajectorySeries,
    net_cx_px: float,
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
    return [h for h in hits if not any(abs(h.frame_idx - bf) <= radius_frames for bf in bounce_frames)]


def detect_net_events(
    traj: TrajectorySeries,
    hits: list[HitEvent],
    bounces: list[BounceEvent],
    net_cx_px: float,
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
        segment_bounces = [b for b in bounces if hit.frame_idx < b.frame_idx < next_hit_frame]
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

        crossed = (
            bool(np.max(seg_x) >= net_cx_px + cross_margin)
            if hit.side == "left"
            else bool(np.min(seg_x) <= net_cx_px - cross_margin)
        )

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
        candidate_near_net = near_dist <= net_near_px and y_ok

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

        segment_bounces = [b for b in bounces if hit.frame_idx < b.frame_idx < next_hit_frame]
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
    events: list[TrajectoryEvent] = []
    for h in hits:
        events.append(TrajectoryEvent(frame_idx=h.frame_idx, kind="hit", side=h.side, cx=h.cx, cy=h.cy, confidence=h.confidence))
    for b in bounces:
        events.append(TrajectoryEvent(frame_idx=b.frame_idx, kind="bounce", side=b.side, cx=b.cx, cy=b.cy, confidence=b.confidence))
    for n in nets:
        events.append(TrajectoryEvent(frame_idx=n.frame_idx, kind="net", side=n.side, cx=n.cx, cy=n.cy, confidence=n.confidence))
    for m in misses:
        events.append(TrajectoryEvent(frame_idx=m.frame_idx, kind="miss", side=m.side, cx=m.cx, cy=m.cy, confidence=m.confidence))

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
_COLOR_NET_EVENT: tuple[int, int, int] = (0, 80, 230)
_COLOR_MISS: tuple[int, int, int] = (40, 80, 255)
_COLOR_NET_LINE: tuple[int, int, int] = (180, 180, 255)
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


def _draw_net_line(frame: np.ndarray, net_cx_px: float) -> None:
    x = int(round(net_cx_px))
    overlay = frame.copy()
    cv2.line(overlay, (x, 0), (x, TARGET_H - 1), _COLOR_NET_LINE, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


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
    overlay = frame.copy()
    cv2.circle(overlay, (int(h.cx), int(h.cy)), max(5, int(12 * alpha)), _COLOR_HIT, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)


def _draw_net_marker(frame: np.ndarray, n: NetEvent, current_frame: int, fade_frames: int = 36) -> None:
    age = current_frame - n.frame_idx
    if age < 0 or age > fade_frames:
        return
    alpha = 1.0 - age / fade_frames
    overlay = frame.copy()
    pt = (int(n.cx), int(n.cy))
    size = max(7, int(14 * alpha))
    cv2.line(overlay, (pt[0] - size, pt[1] - size), (pt[0] + size, pt[1] + size), _COLOR_NET_EVENT, 2, cv2.LINE_AA)
    cv2.line(overlay, (pt[0] - size, pt[1] + size), (pt[0] + size, pt[1] - size), _COLOR_NET_EVENT, 2, cv2.LINE_AA)
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


def _to_heatmap(hm: NDArray[np.float32]) -> np.ndarray:
    return cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)  # type: ignore[return-value]


def _extract_point_events(timeline: dict[int, FrameScore]) -> list[tuple[int, Side]]:
    return [
        (frame_idx, fs.point_winner)
        for frame_idx, fs in sorted(timeline.items())
        if fs.point_winner is not None
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
    return sample_rate, pcm.astype(np.float32) / 32768.0


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
        phrase_audio: dict[Side, NDArray[np.float32]] = {}
        sample_rate = 22_050

        for side, text in phrases.items():
            aiff_path = tmp_dir / f"{side}.aiff"
            wav_path = tmp_dir / f"{side}.wav"
            subprocess.run(["say", "-v", _RUSSIAN_VOICE, "-o", str(aiff_path), text], check=True, capture_output=True, text=True)
            subprocess.run(["ffmpeg", "-y", "-i", str(aiff_path), "-ac", "1", "-ar", str(sample_rate), str(wav_path)], check=True, capture_output=True, text=True)
            sample_rate, phrase_audio[side] = _read_wav_pcm(wav_path)

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
        wav_path = Path(tmp_dir_str) / "score_audio.wav"
        created = _synthesize_score_audio(point_events, fps, total_frames, wav_path)
        if not created:
            silent_video.replace(output_video)
            return
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(silent_video), "-i", str(wav_path), "-c:v", "copy", "-c:a", "aac", "-shortest", str(output_video)],
            check=True,
            capture_output=True,
            text=True,
        )
    silent_video.unlink(missing_ok=True)


def run(video_path: Path, output: Path) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    tracknet = _load_tracknet(TRACKNET_WEIGHTS, device)
    net_cx_px = NET_CX_RATIO * TARGET_W

    en_model: HeatmapEventNet | None = None
    if EVENTNET_WEIGHTS.exists():
        en_model = _load_eventnet(EVENTNET_WEIGHTS, device)
        print(f"EventNet: {EVENTNET_WEIGHTS.name}")
    else:
        print("EventNet: not found — trajectory bounces only")

    seg_model: YOLO | None = None
    if SEG_WEIGHTS.exists():
        seg_model = _load_seg_model(SEG_WEIGHTS)
        print(f"Seg model: {SEG_WEIGHTS.name}")
    else:
        print(f"Seg model: {SEG_WEIGHTS} not found — no segmentation overlay")

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # --- Pass 1: streaming TrackNet ---
    print(f"Pass 1 — TrackNet on {total} frames …")
    detections: dict[int, tuple[float, float]] = {}
    hm_cache: dict[int, NDArray[np.float32]] = {}
    frame_buffer: list[NDArray[np.float32]] = []

    cap = cv2.VideoCapture(str(video_path))
    for fi in tqdm(range(total), desc="TrackNet", unit="frame"):
        ret, raw = cap.read()
        if not ret:
            break

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

    # --- Event detection ---
    print("Detecting events …")
    traj = build_trajectory_series(all_positions)
    if traj is None:
        raise RuntimeError("Not enough tracked points to score the video.")

    traj_bounces = detect_bounces(traj, net_cx_px)

    if en_model is not None:
        en_bounces = _run_eventnet_causal(en_model, hm_cache, all_positions, net_cx_px, device)
        print(f"  EventNet bounces: {len(en_bounces)}  traj bounces: {len(traj_bounces)}")
        bounces = en_bounces if en_bounces else traj_bounces
    else:
        bounces = traj_bounces
        print(f"  traj bounces: {len(bounces)}")

    print(f"  dx range: [{traj.dx.min():.2f}, {traj.dx.max():.2f}]  net_cx_px: {net_cx_px:.1f}")

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

    # --- Scoring ---
    timeline = build_score_timeline(events, detections, total)
    final = timeline.get(total - 1, FrameScore())
    print(f"  Final score: LEFT {final.left} : RIGHT {final.right}")

    # --- Pass 2: render ---
    print("Pass 2 — rendering …")
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

        if seg_model is not None:
            table_mask, person_mask = _predict_seg_masks(seg_model, left)
            if table_mask is not None:
                _draw_table_mask(left, table_mask)
            if person_mask is not None:
                _draw_person_mask(left, person_mask)

        _draw_net_line(left, net_cx_px)
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
