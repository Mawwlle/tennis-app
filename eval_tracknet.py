"""Evaluate TrackNet metrics on an annotated OpenTTGames video."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from tracknet.dataset import TARGET_H, TARGET_W
from tracknet.heatmap import make_empty_heatmap, make_heatmap
from tracknet.model import TrackNet


@dataclass(frozen=True)
class BallSample:
    frame_idx: int
    cx: float
    cy: float
    visible: bool


@dataclass(frozen=True)
class EvalMetrics:
    samples: int
    visible_samples: int
    absent_samples: int
    tp: int
    fp: int
    tn: int
    fn: int
    heatmap_sse: float
    visible_heatmap_sse: float
    absent_heatmap_sse: float
    coord_sse_px2: float
    pixels_per_sample: int
    conf_threshold: float
    dist_tol_px: float

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.samples if self.samples else 0.0

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

    @property
    def heatmap_mse(self) -> float:
        return self.heatmap_sse / (self.samples * self.pixels_per_sample) if self.samples else 0.0

    @property
    def visible_heatmap_mse(self) -> float:
        denom = self.visible_samples * self.pixels_per_sample
        return self.visible_heatmap_sse / denom if denom else 0.0

    @property
    def absent_heatmap_mse(self) -> float:
        denom = self.absent_samples * self.pixels_per_sample
        return self.absent_heatmap_sse / denom if denom else 0.0

    @property
    def coord_mse_px2(self) -> float:
        return self.coord_sse_px2 / self.visible_samples if self.visible_samples else 0.0

    @property
    def coord_rmse_px(self) -> float:
        return math.sqrt(self.coord_mse_px2)


class FrameReader:
    """Small sequential frame reader with a tiny cache for t-2/t-1/t windows."""

    def __init__(self, video_path: Path) -> None:
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_idx = 0
        self._cache: dict[int, np.ndarray] = {}

    def close(self) -> None:
        self._cap.release()

    def get(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(0, frame_idx)
        if frame_idx in self._cache:
            return self._cache[frame_idx]

        if frame_idx < self._next_idx or frame_idx - self._next_idx > 150:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._next_idx = frame_idx

        frame: np.ndarray | None = None
        while self._next_idx <= frame_idx:
            ok, raw = self._cap.read()
            if not ok:
                frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.float32)
            else:
                frame = cv2.resize(raw, (TARGET_W, TARGET_H)).astype(np.float32) / 255.0
            self._cache[self._next_idx] = frame
            self._next_idx += 1

        min_keep = frame_idx - 4
        for old_idx in [idx for idx in self._cache if idx < min_keep]:
            del self._cache[old_idx]

        return self._cache[frame_idx]


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )


def _load_model(weights_path: Path, device: torch.device) -> TrackNet:
    model = TrackNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_samples(markup_path: Path, frame_count: int) -> list[BallSample]:
    raw: dict[str, dict[str, int]] = json.loads(markup_path.read_text())
    samples: list[BallSample] = []
    for frame_str, coords in raw.items():
        frame_idx = int(frame_str)
        if frame_idx >= frame_count:
            continue
        x = float(coords["x"])
        y = float(coords["y"])
        visible = x >= 0 and y >= 0
        samples.append(BallSample(frame_idx, x, y, visible))
    return sorted(samples, key=lambda s: s.frame_idx)


def _frames_tensor(reader: FrameReader, sample: BallSample) -> torch.Tensor:
    frames = [
        reader.get(max(0, sample.frame_idx - 2)),
        reader.get(max(0, sample.frame_idx - 1)),
        reader.get(sample.frame_idx),
    ]
    stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
    return torch.from_numpy(stacked.astype(np.float32))


def _target_heatmap(sample: BallSample, orig_w: int, orig_h: int) -> np.ndarray:
    if not sample.visible:
        return make_empty_heatmap(TARGET_W, TARGET_H)
    cx = sample.cx * TARGET_W / orig_w
    cy = sample.cy * TARGET_H / orig_h
    return make_heatmap(cx, cy, TARGET_W, TARGET_H)


@torch.no_grad()
def _evaluate(
    model: TrackNet,
    reader: FrameReader,
    samples: list[BallSample],
    device: torch.device,
    batch_size: int,
    conf_threshold: float,
    dist_tol_px: float,
) -> EvalMetrics:
    heatmap_sse = 0.0
    visible_heatmap_sse = 0.0
    absent_heatmap_sse = 0.0
    coord_sse_px2 = 0.0
    visible_samples = 0
    absent_samples = 0
    tp = fp = tn = fn = 0

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        frames = torch.stack([_frames_tensor(reader, sample) for sample in batch_samples]).to(device)
        targets_np = np.stack(
            [_target_heatmap(sample, reader.orig_w, reader.orig_h) for sample in batch_samples],
            axis=0,
        )
        targets = torch.from_numpy(targets_np).unsqueeze(1).to(device)

        probs = torch.sigmoid(model(frames))
        sq_error = (probs - targets).square()
        per_sample_sse = sq_error.flatten(1).sum(dim=1).cpu().numpy()

        pred_maps = probs.squeeze(1).cpu().numpy()
        for sample, pred_map, sample_sse in zip(batch_samples, pred_maps, per_sample_sse):
            heatmap_sse += float(sample_sse)
            pred_conf = float(pred_map.max())
            pred_positive = pred_conf >= conf_threshold
            pred_y, pred_x = divmod(int(pred_map.argmax()), TARGET_W)

            if sample.visible:
                visible_samples += 1
                visible_heatmap_sse += float(sample_sse)
                gt_x = sample.cx * TARGET_W / reader.orig_w
                gt_y = sample.cy * TARGET_H / reader.orig_h
                coord_err_px = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                coord_sse_px2 += coord_err_px ** 2
                if pred_positive and coord_err_px <= dist_tol_px:
                    tp += 1
                elif pred_positive:
                    fp += 1
                    fn += 1
                else:
                    fn += 1
            else:
                absent_samples += 1
                absent_heatmap_sse += float(sample_sse)
                if pred_positive:
                    fp += 1
                else:
                    tn += 1

    return EvalMetrics(
        samples=len(samples),
        visible_samples=visible_samples,
        absent_samples=absent_samples,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        heatmap_sse=heatmap_sse,
        visible_heatmap_sse=visible_heatmap_sse,
        absent_heatmap_sse=absent_heatmap_sse,
        coord_sse_px2=coord_sse_px2,
        pixels_per_sample=TARGET_W * TARGET_H,
        conf_threshold=conf_threshold,
        dist_tol_px=dist_tol_px,
    )


def _csv_row(
    *,
    game: str,
    video: Path,
    weights: Path,
    metrics: EvalMetrics,
) -> dict[str, str | int]:
    return {
        "game": game,
        "video": str(video),
        "weights": str(weights),
        "samples": metrics.samples,
        "visible_samples": metrics.visible_samples,
        "absent_samples": metrics.absent_samples,
        "conf_threshold": f"{metrics.conf_threshold:.4f}",
        "dist_tol_px": f"{metrics.dist_tol_px:.2f}",
        "TP": metrics.tp,
        "FP": metrics.fp,
        "TN": metrics.tn,
        "FN": metrics.fn,
        "accuracy": f"{metrics.accuracy:.4f}",
        "precision": f"{metrics.precision:.4f}",
        "recall": f"{metrics.recall:.4f}",
        "f1": f"{metrics.f1:.4f}",
        "heatmap_mse": f"{metrics.heatmap_mse:.8f}",
        "visible_heatmap_mse": f"{metrics.visible_heatmap_mse:.8f}",
        "absent_heatmap_mse": f"{metrics.absent_heatmap_mse:.8f}",
        "coord_mse_px2": f"{metrics.coord_mse_px2:.2f}",
        "coord_rmse_px": f"{metrics.coord_rmse_px:.2f}",
    }


def _write_csv(path: Path, row: dict[str, str | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _print_metrics(metrics: EvalMetrics) -> None:
    print(f"samples:              {metrics.samples}")
    print(f"visible samples:      {metrics.visible_samples}")
    print(f"absent samples:       {metrics.absent_samples}")
    print(f"conf_threshold:       {metrics.conf_threshold:.4f}")
    print(f"dist_tol_px:          {metrics.dist_tol_px:.2f}")
    print(f"accuracy:             {metrics.accuracy:.4f}")
    print(f"precision:            {metrics.precision:.4f}")
    print(f"recall:               {metrics.recall:.4f}")
    print(f"f1:                   {metrics.f1:.4f}")
    print(f"confusion:            TP={metrics.tp} FP={metrics.fp} TN={metrics.tn} FN={metrics.fn}")
    print(f"heatmap_mse:          {metrics.heatmap_mse:.8f}")
    print(f"visible_heatmap_mse:  {metrics.visible_heatmap_mse:.8f}")
    print(f"absent_heatmap_mse:   {metrics.absent_heatmap_mse:.8f}")
    print(f"coord_mse_px2:        {metrics.coord_mse_px2:.2f}")
    print(f"coord_rmse_px:        {metrics.coord_rmse_px:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="test_3")
    parser.add_argument("--markup", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=Path("test_3.mp4"))
    parser.add_argument("--weights", type=Path, default=Path("weights/tracknet_best.pt"))
    parser.add_argument("--csv", type=Path, default=Path("tracknet_mse_metrics.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--dist-tol-px", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    markup_path = args.markup or Path("dataset/openttgames") / args.game / "ball_markup.json"
    if not markup_path.exists():
        raise FileNotFoundError(f"Ball markup not found: {markup_path}")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.weights.exists():
        raise FileNotFoundError(f"TrackNet weights not found: {args.weights}")

    device = _device(args.device)
    model = _load_model(args.weights, device)
    reader = FrameReader(args.video)
    try:
        samples = _load_samples(markup_path, reader.frame_count)
        metrics = _evaluate(
            model,
            reader,
            samples,
            device,
            args.batch_size,
            args.conf_threshold,
            args.dist_tol_px,
        )
    finally:
        reader.close()

    print(f"weights: {args.weights}")
    print(f"game:    {args.game}")
    print(f"markup:  {markup_path}")
    print(f"video:   {args.video}")
    print(f"device:  {device}")
    _print_metrics(metrics)

    row = _csv_row(game=args.game, video=args.video, weights=args.weights, metrics=metrics)
    _write_csv(args.csv, row)
    print(f"csv:     {args.csv}")


if __name__ == "__main__":
    main()
