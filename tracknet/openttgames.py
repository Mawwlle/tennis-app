"""Convert OpenTTGames ball_markup.json annotations to TrackNet Sample objects.

Only frames that are explicitly annotated in ball_markup.json are used:
- x >= 0 and y >= 0  →  visibility=1 (ball visible)
- x == -1 and y == -1  →  visibility=0 (ball absent / out of frame)

Frames absent from ball_markup are skipped entirely — they are unannotated,
not confirmed-absent, and should not be used as training signal.

Expected dataset layout::

    dataset/openttgames/
        game_1/
            game_1.mp4          ← video (download separately)
            ball_markup.json
            events_markup.json
        game_2/ ...
        test_1/ ...
"""

import json
from pathlib import Path

from tracknet.dataset import Sample


_OPENTTGAMES_W = 1920
_OPENTTGAMES_H = 1080


def _load_ball_markup(path: Path) -> dict[int, tuple[float, float]]:
    """Parse ball_markup.json.

    Returns {frame_idx: (x, y)} for ALL annotated frames,
    using (-1.0, -1.0) as sentinel for absent-ball frames.
    """
    raw: dict[str, dict[str, int]] = json.loads(path.read_text())
    return {
        int(k): (float(v["x"]), float(v["y"]))
        for k, v in raw.items()
    }


def build_openttgames_samples(
    data_dir: Path,
    orig_w: int = _OPENTTGAMES_W,
    orig_h: int = _OPENTTGAMES_H,
) -> list[Sample]:
    """Load all annotated frames from OpenTTGames as TrackNet Samples.

    Only games that have a video file alongside ball_markup.json are included.
    Games without a video are skipped with a warning.

    Args:
        data_dir: Root of the OpenTTGames download
                  (contains game_1/, game_2/, ... sub-directories).
        orig_w:   Video width  (1920 for OpenTTGames).
        orig_h:   Video height (1080 for OpenTTGames).

    Returns:
        List of Sample objects ready for TrackNet training.
    """
    samples: list[Sample] = []
    n_visible = 0
    n_absent  = 0
    n_skipped_no_video = 0

    game_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not game_dirs:
        raise ValueError(f"No game sub-directories found in {data_dir}")

    for game_dir in game_dirs:
        ball_path = game_dir / "ball_markup.json"
        if not ball_path.exists():
            continue

        # Find the video file (mp4 or mov)
        video_files = [
            f for f in game_dir.iterdir()
            if f.suffix.lower() in {".mp4", ".mov", ".avi"}
        ]
        if not video_files:
            print(f"  [skip] {game_dir.name}  — no video file (download with --video flag)")
            n_skipped_no_video += 1
            continue

        video_path = video_files[0]
        frame_map  = _load_ball_markup(ball_path)

        for frame_idx, (x, y) in frame_map.items():
            visible = x >= 0 and y >= 0
            samples.append(Sample(
                video_path=video_path,
                frame_idx=frame_idx,
                cx=x if visible else 0.0,
                cy=y if visible else 0.0,
                orig_w=orig_w,
                orig_h=orig_h,
                visibility=1 if visible else 0,
            ))
            if visible:
                n_visible += 1
            else:
                n_absent += 1

    n_games = len(game_dirs) - n_skipped_no_video
    print(
        f"OpenTTGames TrackNet: {n_games} games, "
        f"{len(samples)} samples "
        f"(visible={n_visible}, absent={n_absent})"
        + (f", {n_skipped_no_video} skipped (no video)" if n_skipped_no_video else "")
    )
    return samples
