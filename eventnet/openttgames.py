"""Parser for OpenTTGames dataset annotations → HeatmapEventSample list.

Expected dataset layout (after download)::

    dataset/openttgames/
        game_1/
            ball_markup.json
            events_markup.json
        game_2/ ...
        test_1/ ...
        ...

JSON formats
------------
ball_markup.json::

    {"0": {"x": 123, "y": 456}, "1": {"x": -1, "y": -1}, ...}

events_markup.json::

    {"42": "bounce", "87": "net", "120": "empty_event", ...}

Label mapping
-------------
- "bounce"      → bounce  (table bounce — clear trajectory reversal)
- "net"         → skipped (net class removed)
- "empty_event" → skipped (ambiguous racket contact)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from eventnet.dataset import LABEL_TO_IDX
from eventnet.heatmap import (
    SMOOTH_RADIUS,
    HeatmapEventSample,
    _smooth_weight,
    positions_to_heatmaps,
)

_FRAME_WIDTH  = 1920.0
_FRAME_HEIGHT = 1080.0

_LABEL_MAP: dict[str, str | None] = {
    "bounce":      "bounce",
    "net":         None,
    "empty_event": None,
}


def _load_ball_markup(path: Path) -> dict[int, tuple[float, float]]:
    """Parse ball_markup.json → {frame_idx: (x_norm, y_norm)}, excluding absent frames."""
    raw: dict[str, dict[str, int]] = json.loads(path.read_text())
    frame_map: dict[int, tuple[float, float]] = {}
    for frame_str, coords in raw.items():
        x, y = coords["x"], coords["y"]
        if x >= 0 and y >= 0:
            frame_map[int(frame_str)] = (float(x) / _FRAME_WIDTH, float(y) / _FRAME_HEIGHT)
    return frame_map


def _load_events_markup(path: Path) -> dict[int, str]:
    raw: dict[str, str] = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def _extract_window_norm(
    frame_map: dict[int, tuple[float, float]],
    center: int,
    half: int,
) -> list[tuple[float, float] | None] | None:
    """Extract window of already-normalized positions. Returns None if >50% missing."""
    n = 2 * half + 1
    window: list[tuple[float, float] | None] = []
    missing = 0
    for idx in range(center - half, center + half + 1):
        pos = frame_map.get(idx)
        window.append(pos)
        if pos is None:
            missing += 1
    if missing > n // 2:
        return None
    return window


def build_openttgames_heatmap_samples(
    data_dir: Path,
    window_size: int = 15,
    smooth_radius: int = SMOOTH_RADIUS,
    neg_ratio: float = 3.0,
    rng_seed: int = 42,
) -> list[HeatmapEventSample]:
    """Load all games and return HeatmapEventSample list with smooth labeling.

    Only bounce events are used (net/empty_event skipped).
    Uses TTNet-style smooth temporal labeling: each event expands to
    (2*smooth_radius+1) samples with decaying sin weights.
    """
    rng  = random.Random(rng_seed)
    half = window_size // 2
    samples: list[HeatmapEventSample] = []
    skipped = 0
    counts: dict[str, int] = {}

    game_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not game_dirs:
        raise ValueError(f"No game sub-directories found in {data_dir}")

    for game_dir in game_dirs:
        ball_path   = game_dir / "ball_markup.json"
        events_path = game_dir / "events_markup.json"
        if not ball_path.exists() or not events_path.exists():
            continue

        frame_map = _load_ball_markup(ball_path)
        events    = _load_events_markup(events_path)

        event_frames: set[int] = set()

        # ── Positive samples with smooth labeling ─────────────────────────────
        for frame_idx, event_str in events.items():
            label = _LABEL_MAP.get(event_str)
            if label is None:
                continue
            label_idx = LABEL_TO_IDX[label]
            event_frames.add(frame_idx)

            for d in range(-smooth_radius, smooth_radius + 1):
                center = frame_idx + d
                window = _extract_window_norm(frame_map, center, half)
                if window is None:
                    skipped += 1
                    continue
                hm = positions_to_heatmaps(tuple(window))
                w  = _smooth_weight(d, smooth_radius)
                samples.append(HeatmapEventSample(heatmaps=hm, label_idx=label_idx, weight=w))
                counts[label] = counts.get(label, 0) + 1

        # ── Negative samples ──────────────────────────────────────────────────
        all_frames = sorted(frame_map.keys())
        candidates = [
            f for f in all_frames
            if all(abs(f - ef) > half + smooth_radius for ef in event_frames)
            and f >= all_frames[0] + half
            and f <= all_frames[-1] - half
        ]
        n_event = sum(1 for ev in events.values() if _LABEL_MAP.get(ev) is not None)
        n_neg   = int(n_event * neg_ratio)
        chosen  = rng.sample(candidates, min(n_neg, len(candidates)))
        for fi in chosen:
            window = _extract_window_norm(frame_map, fi, half)
            if window is None:
                continue
            hm = positions_to_heatmaps(tuple(window))
            samples.append(HeatmapEventSample(
                heatmaps=hm,
                label_idx=LABEL_TO_IDX["none"],
                weight=1.0,
            ))

    raw_events = sum(counts.values()) // (2 * smooth_radius + 1)
    print(
        f"OpenTTGames: {len(game_dirs)} games, "
        f"~{raw_events} events → {sum(counts.values())} augmented samples {counts}, "
        f"{skipped} skipped (missing frames)"
    )
    return samples
