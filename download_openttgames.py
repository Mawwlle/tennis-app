"""Download OpenTTGames dataset (annotations and/or videos).

Usage::

    # Annotations only (~1.5 MB, needed for EventNet / TCN training)
    uv run python download_openttgames.py

    # Annotations + videos (~50 GB, needed for TrackNet training)
    uv run python download_openttgames.py --video

Downloads into dataset/openttgames/{game_N}/.

If the automatic download fails, follow the manual instructions printed at
the end of this script.
"""

import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

# Official dataset page: https://lab.osai.ai/datasets/openttgames/
_BASE_URL = "https://lab.osai.ai/datasets/openttgames"

_TRAINING_GAMES = ["game_1", "game_2", "game_3", "game_4", "game_5"]
_TEST_GAMES     = ["test_1", "test_2", "test_3", "test_4", "test_5", "test_6", "test_7"]
_GAMES          = _TRAINING_GAMES + _TEST_GAMES

_OUTPUT_DIR       = Path("dataset/openttgames")
_ANNOTATION_FILES = {"ball_markup.json", "events_markup.json"}


def _download_stream(url: str, dest: Path, desc: str = "") -> None:
    """Download url to dest, showing progress."""
    label = desc or dest.name
    print(f"  GET {url}")
    with urlopen(url, timeout=120) as resp:
        total = int(resp.getheader("Content-Length") or 0)
        downloaded = 0
        chunk = 1 << 20  # 1 MB
        with dest.open("wb") as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                downloaded += len(block)
                if total:
                    pct = downloaded / total * 100
                    mb  = downloaded / 1e6
                    print(f"\r    {label}  {mb:.0f}/{total/1e6:.0f} MB  ({pct:.0f}%)", end="", flush=True)
        print()


def _download_bytes(url: str) -> bytes:
    print(f"  GET {url}")
    with urlopen(url, timeout=60) as resp:
        return resp.read()


def _extract_annotations(archive_bytes: bytes, dest: Path) -> bool:
    """Extract only JSON annotation files from a ZIP archive."""
    dest.mkdir(parents=True, exist_ok=True)
    extracted = False
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for name in zf.namelist():
            filename = Path(name).name
            if filename in _ANNOTATION_FILES:
                data = zf.read(name)
                (dest / filename).write_bytes(data)
                print(f"    ✓ {filename}  ({len(data):,} bytes)")
                extracted = True
    return extracted


def download_annotations(games: list[str]) -> list[str]:
    """Download JSON annotation archives. Returns list of failed games."""
    failed: list[str] = []
    for game in games:
        dest = _OUTPUT_DIR / game
        if (dest / "ball_markup.json").exists() and (dest / "events_markup.json").exists():
            print(f"[skip] {game}  (annotations already present)")
            continue

        print(f"[{game}] annotations")
        urls = [
            f"{_BASE_URL}/data/{game}.zip",
            f"{_BASE_URL}/{game}.zip",
        ]
        success = False
        for url in urls:
            try:
                data = _download_bytes(url)
                if _extract_annotations(data, dest):
                    success = True
                    break
                else:
                    print("    ✗ archive has no annotation JSONs")
            except URLError as exc:
                print(f"    ✗ {exc}")

        if not success:
            failed.append(game)

    return failed


def download_videos(games: list[str]) -> list[str]:
    """Download MP4 video files. Returns list of failed games."""
    failed: list[str] = []
    for game in games:
        dest_dir  = _OUTPUT_DIR / game
        dest_file = dest_dir / f"{game}.mp4"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if dest_file.exists():
            print(f"[skip] {game}  (video already present)")
            continue

        print(f"[{game}] video")
        urls = [
            f"{_BASE_URL}/data/{game}.mp4",
            f"{_BASE_URL}/{game}.mp4",
        ]
        success = False
        for url in urls:
            try:
                _download_stream(url, dest_file, desc=f"{game}.mp4")
                success = True
                break
            except URLError as exc:
                print(f"    ✗ {exc}")
                if dest_file.exists():
                    dest_file.unlink()

        if not success:
            failed.append(game)

    return failed


def _print_manual_instructions(failed: list[str]) -> None:
    print(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Automatic download failed for: " + ", ".join(failed) + "\n"
        "\n"
        "Manual download instructions\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "1. Go to  https://lab.osai.ai/datasets/openttgames/\n"
        "   or use the Kaggle mirror:\n"
        "   https://www.kaggle.com/datasets/heftyjohnson/openttgames\n"
        "\n"
        "2. For each game, download:\n"
        "   - Annotation archive (ball_markup.json + events_markup.json)\n"
        "   - Video file (MP4, needed only for TrackNet training)\n"
        "\n"
        "3. Place files so that the layout is:\n"
        "   dataset/openttgames/\n"
        "       game_1/ball_markup.json\n"
        "       game_1/events_markup.json\n"
        "       game_1/game_1.mp4          ← only if training TrackNet\n"
        "       game_2/...\n"
        "       test_1/...\n"
        "\n"
        "4. Re-run training:\n"
        "   EventNet (no videos needed):  uv run python train_eventnet.py\n"
        "   TrackNet  (videos needed):    uv run python train_tracknet.py\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def main() -> None:
    want_video = "--video" in sys.argv

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Downloading annotations ===")
    failed_ann = download_annotations(_GAMES)

    failed_vid: list[str] = []
    if want_video:
        print("\n=== Downloading videos (training games only, ~50 GB) ===")
        failed_vid = download_videos(_TRAINING_GAMES)

    failed = failed_ann + failed_vid
    if failed:
        _print_manual_instructions(failed)
        sys.exit(1)

    total_ann = sum(
        1 for g in _GAMES for f in _ANNOTATION_FILES
        if (_OUTPUT_DIR / g / f).exists()
    )
    total_vid = sum(
        1 for g in _TRAINING_GAMES
        if (_OUTPUT_DIR / g / f"{g}.mp4").exists()
    )
    print(f"\nDone.  {total_ann} annotation files,  {total_vid} videos  →  {_OUTPUT_DIR}")

    if not want_video and total_vid < len(_TRAINING_GAMES):
        print(
            "\nNote: videos not downloaded yet.\n"
            "      Run with --video to download them for TrackNet training:\n"
            "      uv run python download_openttgames.py --video"
        )


if __name__ == "__main__":
    main()
