"""Compatibility wrapper for the old YOLO training command.

New explicit commands:
  uv run prepare-yolo-dataset
  uv run train-det
"""

from __future__ import annotations

import sys

from prepare_yolo_dataset import main as prepare_main
from train_det import main as train_main

_TRAIN_FLAGS = {
    "--data",
    "--base-model",
    "--output-weights",
    "--run-name",
    "--epochs",
    "--batch",
    "--imgsz",
    "--device",
}


def main() -> None:
    if "--train-only" in sys.argv:
        sys.argv.remove("--train-only")
        train_main()
        return

    if "--prepare-only" in sys.argv:
        sys.argv.remove("--prepare-only")
        prepare_main()
        return

    if any(arg in _TRAIN_FLAGS for arg in sys.argv[1:]):
        train_main()
        return

    prepare_main()
    train_main()


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    main()
