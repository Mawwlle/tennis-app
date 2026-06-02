"""Evaluate HeatmapEventNet on an OpenTTGames game with a full-frame scan.

Reports binary bounce-vs-none metrics using the same argmax decision rule as
the inference script, and saves a compact CSV summary.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eventnet.dataset import IDX_TO_LABEL, LABEL_TO_IDX
from eventnet.heatmap import (
    SMOOTH_RADIUS,
    HeatmapEventDataset,
    HeatmapEventSample,
    positions_to_heatmaps,
)
from eventnet.model import HeatmapEventNet, HeatmapEventNetConfig
from eventnet.openttgames import (
    _extract_window_norm,
    _load_ball_markup,
    _load_events_markup,
)


@dataclass(frozen=True)
class EvalMetrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )


def _load_model(weights_path: Path, device: torch.device) -> tuple[HeatmapEventNet, HeatmapEventNetConfig]:
    checkpoint = torch.load(weights_path, map_location=device)
    cfg = HeatmapEventNetConfig(**checkpoint["cfg"])
    model = HeatmapEventNet(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, cfg


def _build_full_scan_samples(
    game_dir: Path,
    window_size: int,
    smooth_radius: int,
) -> tuple[list[HeatmapEventSample], int, int]:
    """Build one sample for every valid frame center in a game."""
    half = window_size // 2
    samples: list[HeatmapEventSample] = []
    skipped = 0

    frame_map = _load_ball_markup(game_dir / "ball_markup.json")
    events = _load_events_markup(game_dir / "events_markup.json")
    bounce_frames = {
        frame_idx
        for frame_idx, event_str in events.items()
        if event_str == "bounce"
    }

    all_frames = sorted(frame_map)
    for center in range(all_frames[0] + half, all_frames[-1] - half + 1):
        window = _extract_window_norm(frame_map, center, half)
        if window is None:
            skipped += 1
            continue

        is_bounce = any(abs(center - frame_idx) <= smooth_radius for frame_idx in bounce_frames)
        label_idx = LABEL_TO_IDX["bounce"] if is_bounce else LABEL_TO_IDX["none"]
        samples.append(
            HeatmapEventSample(
                heatmaps=positions_to_heatmaps(tuple(window)),
                label_idx=label_idx,
                weight=1.0,
            )
        )

    return samples, skipped, len(bounce_frames)


@torch.no_grad()
def _evaluate_samples(
    model: HeatmapEventNet,
    samples: list[HeatmapEventSample],
    device: torch.device,
    batch_size: int,
) -> EvalMetrics:
    dataset = HeatmapEventDataset(samples, dropout_p=0.0)
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    tp = fp = tn = fn = 0
    positive_idx = LABEL_TO_IDX["bounce"]

    for heatmaps, targets, _weights in loader:
        heatmaps = heatmaps.to(device)
        logits = model(heatmaps)
        pred_cls = logits.argmax(dim=1).cpu()
        target_cls = targets.argmax(dim=1)

        pred_pos = pred_cls == positive_idx
        target_pos = target_cls == positive_idx
        tp += int((pred_pos & target_pos).sum().item())
        fp += int((pred_pos & ~target_pos).sum().item())
        tn += int((~pred_pos & ~target_pos).sum().item())
        fn += int((~pred_pos & target_pos).sum().item())

    return EvalMetrics(n=len(samples), tp=tp, fp=fp, tn=tn, fn=fn)


def _print_metrics(name: str, metrics: EvalMetrics) -> None:
    print(f"\n{name}")
    print(f"  samples:   {metrics.n}")
    print(f"  accuracy:  {metrics.accuracy:.4f}")
    print(f"  precision: {metrics.precision:.4f}")
    print(f"  recall:    {metrics.recall:.4f}")
    print(f"  f1:        {metrics.f1:.4f}")
    print(
        "  confusion: "
        f"TP={metrics.tp} FP={metrics.fp} TN={metrics.tn} FN={metrics.fn} "
        f"(positive={IDX_TO_LABEL[LABEL_TO_IDX['bounce']]})"
    )


def _csv_row(
    *,
    game: str,
    weights: Path,
    smooth_radius: int,
    bounce_events: int,
    skipped_centers: int,
    metrics: EvalMetrics,
) -> dict[str, str | int]:
    return {
        "game": game,
        "weights": str(weights),
        "smooth_radius": smooth_radius,
        "bounce_events": bounce_events,
        "samples": metrics.n,
        "skipped_centers": skipped_centers,
        "TP": metrics.tp,
        "FP": metrics.fp,
        "TN": metrics.tn,
        "FN": metrics.fn,
        "accuracy": f"{metrics.accuracy:.4f}",
        "precision": f"{metrics.precision:.4f}",
        "recall": f"{metrics.recall:.4f}",
        "f1": f"{metrics.f1:.4f}",
    }


def _write_csv(path: Path, row: dict[str, str | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/openttgames"))
    parser.add_argument("--game", default="test_2")
    parser.add_argument("--weights", type=Path, default=Path("weights/eventnet_best.pt"))
    parser.add_argument("--csv", type=Path, default=Path("eventnet_full_scan_metrics.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smooth-radius", type=int, default=SMOOTH_RADIUS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    game_dir = args.data_dir / args.game
    if not game_dir.exists():
        raise FileNotFoundError(f"Game directory not found: {game_dir}")
    if not args.weights.exists():
        raise FileNotFoundError(f"EventNet weights not found: {args.weights}")

    device = _device(args.device)
    model, cfg = _load_model(args.weights, device)

    print(f"weights: {args.weights}")
    print(f"game:    {game_dir}")
    print(f"device:  {device}")
    print(f"classes: {IDX_TO_LABEL}")

    full_scan_samples, full_scan_skipped, n_bounces = _build_full_scan_samples(
        game_dir=game_dir,
        window_size=cfg.window_size,
        smooth_radius=args.smooth_radius,
    )
    print(f"bounce events: {n_bounces}")
    print(f"full-scan skipped centers: {full_scan_skipped}")
    full_scan_metrics = _evaluate_samples(model, full_scan_samples, device, args.batch_size)
    _print_metrics(
        f"Full scan over valid centers (bounce target within ±{args.smooth_radius} frames)",
        full_scan_metrics,
    )

    row = _csv_row(
        game=args.game,
        weights=args.weights,
        smooth_radius=args.smooth_radius,
        bounce_events=n_bounces,
        skipped_centers=full_scan_skipped,
        metrics=full_scan_metrics,
    )
    _write_csv(args.csv, row)
    print(f"\ncsv: {args.csv}")


if __name__ == "__main__":
    main()
