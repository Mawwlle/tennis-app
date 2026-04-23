"""Download OpenTTGames dataset (annotations, segmentation masks and/or videos).

Usage::

    # JSON annotations only (~1.5 MB, needed for EventNet / TCN training)
    uv run python download_openttgames.py

    # JSON annotations + segmentation masks (needed for unified segmentation)
    uv run python download_openttgames.py --masks

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

TRAINING_GAMES = ["game_1", "game_2", "game_3", "game_4", "game_5"]
TEST_GAMES     = ["test_1", "test_2", "test_3", "test_4", "test_5", "test_6", "test_7"]
GAMES          = TRAINING_GAMES + TEST_GAMES

OUTPUT_DIR        = Path("dataset/openttgames")
ANNOTATION_FILES  = {"ball_markup.json", "events_markup.json"}
SEGMENT_MASK_GLOB = "*.png"

_TRAINING_GAMES = TRAINING_GAMES
_TEST_GAMES = TEST_GAMES
_GAMES = GAMES
_OUTPUT_DIR = OUTPUT_DIR
_ANNOTATION_FILES = ANNOTATION_FILES


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


def _extract_markup(archive_bytes: bytes, dest: Path, include_masks: bool) -> bool:
    """Extract JSON annotation files and optionally segmentation masks."""
    dest.mkdir(parents=True, exist_ok=True)
    extracted = False
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for name in zf.namelist():
            filename = Path(name).name
            if filename in ANNOTATION_FILES:
                data = zf.read(name)
                (dest / filename).write_bytes(data)
                print(f"    ✓ {filename}  ({len(data):,} bytes)")
                extracted = True
                continue

            if include_masks and filename.lower().endswith(".png"):
                data = zf.read(name)
                (dest / filename).write_bytes(data)
                extracted = True
    return extracted


def _has_required_markup(dest: Path, include_masks: bool) -> bool:
    have_json = all((dest / filename).exists() for filename in ANNOTATION_FILES)
    if not have_json:
        return False
    if not include_masks:
        return True
    return any(dest.glob(SEGMENT_MASK_GLOB))


def download_annotations(games: list[str], include_masks: bool = False) -> list[str]:
    """Download JSON annotations and optional segmentation masks."""
    failed: list[str] = []
    for game in games:
        dest = OUTPUT_DIR / game
        if _has_required_markup(dest, include_masks):
            suffix = " + masks" if include_masks else ""
            print(f"[skip] {game}  (annotations{suffix} already present)")
            continue

        print(f"[{game}] annotations{' + masks' if include_masks else ''}")
        urls = [
            f"{_BASE_URL}/data/{game}.zip",
            f"{_BASE_URL}/{game}.zip",
        ]
        success = False
        for url in urls:
            try:
                data = _download_bytes(url)
                if _extract_markup(data, dest, include_masks=include_masks):
                    success = True
                    break
                else:
                    print("    ✗ archive has no required markup files")
            except URLError as exc:
                print(f"    ✗ {exc}")

        if not success:
            failed.append(game)

    return failed


def download_videos(games: list[str]) -> list[str]:
    """Download MP4 video files. Returns list of failed games."""
    failed: list[str] = []
    for game in games:
        dest_dir  = OUTPUT_DIR / game
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
        "   - Segmentation masks (.png files from the annotation ZIP, needed for train-seg)\n"
        "   - Video file (MP4, needed only for TrackNet training)\n"
        "\n"
        "3. Place files so that the layout is:\n"
        "   dataset/openttgames/\n"
        "       game_1/ball_markup.json\n"
        "       game_1/events_markup.json\n"
        "       game_1/000123.png           ← segmentation mask frame\n"
        "       game_1/game_1.mp4          ← only if training TrackNet\n"
        "       game_2/...\n"
        "       test_1/...\n"
        "\n"
        "4. Re-run training:\n"
        "   EventNet (no videos needed):  uv run python train_eventnet.py\n"
        "   TrackNet  (videos needed):    uv run python train_tracknet.py\n"
        "   Unified seg (masks + train videos):  uv run train-seg\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def main() -> None:
    want_video = "--video" in sys.argv
    want_masks = "--masks" in sys.argv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Downloading annotations{' + masks' if want_masks else ''} ===")
    failed_ann = download_annotations(GAMES, include_masks=want_masks)

    failed_vid: list[str] = []
    if want_video:
        print("\n=== Downloading videos (training games only, ~50 GB) ===")
        failed_vid = download_videos(TRAINING_GAMES)

    failed = failed_ann + failed_vid
    if failed:
        _print_manual_instructions(failed)
        sys.exit(1)

    total_ann = sum(
        1 for g in GAMES for f in ANNOTATION_FILES
        if (OUTPUT_DIR / g / f).exists()
    )
    total_masks = sum(
        sum(1 for _ in (OUTPUT_DIR / g).glob(SEGMENT_MASK_GLOB))
        for g in GAMES
    )
    total_vid = sum(
        1 for g in TRAINING_GAMES
        if (OUTPUT_DIR / g / f"{g}.mp4").exists()
    )
    print(f"\nDone.  {total_ann} annotation files,  {total_masks} masks,  {total_vid} videos  →  {OUTPUT_DIR}")

    if not want_video and total_vid < len(TRAINING_GAMES):
        print(
            "\nNote: videos not downloaded yet.\n"
            "      Run with --video to download them for TrackNet training:\n"
            "      uv run python download_openttgames.py --video"
        )

    if not want_masks and total_masks == 0:
        print(
            "\nNote: segmentation masks not downloaded yet.\n"
            "      Run with --masks to download them for unified segmentation training:\n"
            "      uv run python download_openttgames.py --masks"
        )


if __name__ == "__main__":
    main()
