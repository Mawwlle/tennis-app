"""Parser for OpenTTGames dataset annotations → EventSample list.

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
- "bounce"      → bounce
- "net"         → net
- "empty_event" → skipped by default (racket contact near table, no clear trajectory change)
"""

import json
from pathlib import Path

from eventnet.dataset import LABEL_TO_IDX, EventSample, _extract_window


_FRAME_WIDTH  = 1920.0
_FRAME_HEIGHT = 1080.0

_LABEL_MAP: dict[str, str | None] = {
    "bounce":      "bounce",
    "net":         "net",
    "empty_event": None,    # skipped — ambiguous label (racket contact w/o clear physics change)
}


def _load_ball_markup(path: Path) -> dict[int, tuple[float, float]]:
    """Parse ball_markup.json → {frame_idx: (x, y)}, excluding absent frames."""
    raw: dict[str, dict[str, int]] = json.loads(path.read_text())
    frame_map: dict[int, tuple[float, float]] = {}
    for frame_str, coords in raw.items():
        x, y = coords["x"], coords["y"]
        if x >= 0 and y >= 0:
            frame_map[int(frame_str)] = (float(x), float(y))
    return frame_map


def _load_events_markup(path: Path) -> dict[int, str]:
    """Parse events_markup.json → {frame_idx: event_string}."""
    raw: dict[str, str] = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def build_openttgames_samples(
    data_dir: Path,
    window_size: int = 9,
    frame_width: float = _FRAME_WIDTH,
    frame_height: float = _FRAME_HEIGHT,
) -> list[EventSample]:
    """Load all games under data_dir and return EventSample list.

    Only bounce and net events are included (empty_event is skipped).
    Windows where >50% of frames are missing are also discarded.

    Args:
        data_dir:     Root of the OpenTTGames download
                      (contains game_1/, game_2/, ... sub-directories).
        window_size:  Must match the window_size used in the main dataset.
        frame_width:  Video width in pixels (1920 for OpenTTGames).
        frame_height: Video height in pixels (1080 for OpenTTGames).

    Returns:
        List of EventSample with label_idx in {bounce=1, net=2}.
    """
    half    = window_size // 2
    samples: list[EventSample] = []
    skipped_missing = 0
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

        for frame_idx, event_str in events.items():
            label = _LABEL_MAP.get(event_str)
            if label is None:
                continue

            window = _extract_window(frame_map, frame_idx, half, frame_width, frame_height)
            if window is None:
                skipped_missing += 1
                continue

            samples.append(EventSample(positions=tuple(window), label_idx=LABEL_TO_IDX[label]))
            counts[label] = counts.get(label, 0) + 1

    print(
        f"OpenTTGames: {len(game_dirs)} games, "
        f"{sum(counts.values())} samples {counts}, "
        f"{skipped_missing} skipped (missing frames)"
    )
    return samples
