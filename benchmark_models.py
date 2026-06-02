"""Benchmark TrackNet vs YOLO ONNX latency stability and memory usage.

The benchmark measures model inference latency on preprocessed video frames.
It saves raw per-frame measurements, summary statistics, and plots with
mean +/- std bands across repeated runs.

Usage:
    uv run python benchmark_models.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import psutil
from numpy.typing import NDArray


DEFAULT_VIDEO = Path("dataset/videos/night_videos/part_1.mp4")
DEFAULT_OUTPUT_DIR = Path("benchmark_results")
TRACKNET_ONNX = Path("weights/tracknet.onnx")
YOLO_ONNX = Path("weights/yolo_det.onnx")

TRACKNET_W = 640
TRACKNET_H = 360
YOLO_SIZE = 640


@dataclass(frozen=True)
class ModelSummary:
    model: str
    runs: int
    frames_per_run: int
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_rss_mb: float
    std_rss_mb: float
    peak_rss_mb: float
    mean_rss_delta_mb: float
    std_rss_delta_mb: float
    peak_rss_delta_mb: float
    session_load_rss_mb: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracknet", type=Path, default=TRACKNET_ONNX)
    parser.add_argument("--yolo", type=Path, default=YOLO_ONNX)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime provider. Defaults to CPUExecutionProvider.",
    )
    return parser.parse_args()


def _rss_mb(process: psutil.Process) -> float:
    return process.memory_info().rss / (1024 * 1024)


def _load_video_frames(video_path: Path, frame_count: int) -> list[NDArray[np.uint8]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[NDArray[np.uint8]] = []
    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from: {video_path}")

    if len(frames) < frame_count:
        repeats = int(np.ceil(frame_count / len(frames)))
        frames = (frames * repeats)[:frame_count]

    return frames


def _prepare_tracknet_inputs(frames: list[NDArray[np.uint8]]) -> list[NDArray[np.float32]]:
    small_frames = [
        cv2.resize(frame, (TRACKNET_W, TRACKNET_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        for frame in frames
    ]
    inputs: list[NDArray[np.float32]] = []
    for idx in range(len(small_frames)):
        window = [
            small_frames[max(0, idx - 2)],
            small_frames[max(0, idx - 1)],
            small_frames[idx],
        ]
        stacked = np.concatenate([frame.transpose(2, 0, 1) for frame in window], axis=0)
        inputs.append(np.expand_dims(np.ascontiguousarray(stacked), axis=0))
    return inputs


def _prepare_yolo_inputs(frames: list[NDArray[np.uint8]]) -> list[NDArray[np.float32]]:
    inputs: list[NDArray[np.float32]] = []
    for frame in frames:
        resized = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        inputs.append(np.expand_dims(np.ascontiguousarray(chw), axis=0))
    return inputs


def _session(model_path: Path, provider: str) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if provider not in ort.get_available_providers():
        raise RuntimeError(
            f"Provider {provider!r} is unavailable. Available: {ort.get_available_providers()}"
        )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=[provider])


def _run_model(
    *,
    name: str,
    model_path: Path,
    inputs: list[NDArray[np.float32]],
    runs: int,
    warmup: int,
    provider: str,
    process: psutil.Process,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], ModelSummary]:
    gc.collect()
    rss_before = _rss_mb(process)
    session = _session(model_path, provider)
    input_name = session.get_inputs()[0].name
    rss_after_load = _rss_mb(process)

    for idx in range(min(warmup, len(inputs))):
        session.run(None, {input_name: inputs[idx]})

    latency_rows: list[dict[str, float | int | str]] = []
    memory_rows: list[dict[str, float | int | str]] = []

    for run_idx in range(runs):
        gc.collect()
        for frame_idx, tensor in enumerate(inputs):
            start = time.perf_counter_ns()
            session.run(None, {input_name: tensor})
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            rss = _rss_mb(process)

            latency_rows.append(
                {
                    "model": name,
                    "run": run_idx,
                    "frame": frame_idx,
                    "latency_ms": elapsed_ms,
                }
            )
            memory_rows.append(
                {
                    "model": name,
                    "run": run_idx,
                    "frame": frame_idx,
                    "rss_mb": rss,
                    "rss_delta_from_before_load_mb": rss - rss_before,
                }
            )

    latencies = np.array([float(row["latency_ms"]) for row in latency_rows], dtype=np.float64)
    rss_values = np.array([float(row["rss_mb"]) for row in memory_rows], dtype=np.float64)
    rss_deltas = np.array(
        [float(row["rss_delta_from_before_load_mb"]) for row in memory_rows],
        dtype=np.float64,
    )
    summary = ModelSummary(
        model=name,
        runs=runs,
        frames_per_run=len(inputs),
        mean_latency_ms=float(latencies.mean()),
        std_latency_ms=float(latencies.std(ddof=1)),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p90_latency_ms=float(np.percentile(latencies, 90)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        min_latency_ms=float(latencies.min()),
        max_latency_ms=float(latencies.max()),
        mean_rss_mb=float(rss_values.mean()),
        std_rss_mb=float(rss_values.std(ddof=1)),
        peak_rss_mb=float(rss_values.max()),
        mean_rss_delta_mb=float(rss_deltas.mean()),
        std_rss_delta_mb=float(rss_deltas.std(ddof=1)),
        peak_rss_delta_mb=float(rss_deltas.max()),
        session_load_rss_mb=float(rss_after_load - rss_before),
    )
    return latency_rows, memory_rows, summary


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in keys) + "\n")


def _series_stats(
    rows: list[dict[str, float | int | str]],
    value_key: str,
    model: str,
    frames: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    values_by_frame: list[list[float]] = [[] for _ in range(frames)]
    for row in rows:
        if row["model"] != model:
            continue
        values_by_frame[int(row["frame"])].append(float(row[value_key]))

    x = np.arange(frames)
    mean = np.array([np.mean(v) for v in values_by_frame], dtype=np.float64)
    std = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in values_by_frame], dtype=np.float64)
    return x, mean, std


def _plot_latency(
    path: Path,
    rows: list[dict[str, float | int | str]],
    summaries: list[ModelSummary],
    frames: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for summary in summaries:
        x, mean, std = _series_stats(rows, "latency_ms", summary.model, frames)
        ax.plot(x, mean, label=f"{summary.model} mean {summary.mean_latency_ms:.2f} ms")
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, label=f"{summary.model} +/- std {summary.std_latency_ms:.2f} ms")

    ax.set_title("Latency over time: mean +/- std across runs")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Inference latency, ms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_memory(
    path: Path,
    rows: list[dict[str, float | int | str]],
    summaries: list[ModelSummary],
    frames: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for summary in summaries:
        x, mean, std = _series_stats(rows, "rss_mb", summary.model, frames)
        ax.plot(x, mean, label=f"{summary.model} mean {summary.mean_rss_mb:.1f} MB")
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, label=f"{summary.model} +/- std {summary.std_rss_mb:.1f} MB")

    ax.set_title("Process RSS over time: mean +/- std across runs")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("RSS, MB")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_memory_delta(
    path: Path,
    rows: list[dict[str, float | int | str]],
    summaries: list[ModelSummary],
    frames: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for summary in summaries:
        x, mean, std = _series_stats(rows, "rss_delta_from_before_load_mb", summary.model, frames)
        ax.plot(x, mean, label=f"{summary.model} mean +{summary.mean_rss_delta_mb:.1f} MB")
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, label=f"{summary.model} +/- std {summary.std_rss_delta_mb:.1f} MB")

    ax.set_title("Additional process RSS over time: mean +/- std across runs")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("RSS delta from before model load, MB")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_distribution(
    path: Path,
    rows: list[dict[str, float | int | str]],
    summaries: list[ModelSummary],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for summary in summaries:
        values = [float(row["latency_ms"]) for row in rows if row["model"] == summary.model]
        ax.hist(values, bins=50, alpha=0.45, label=f"{summary.model}: mean {summary.mean_latency_ms:.2f}, std {summary.std_latency_ms:.2f} ms")

    ax.set_title("Latency distribution across all runs")
    ax.set_xlabel("Inference latency, ms")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process(os.getpid())
    frames = _load_video_frames(args.video, args.frames)
    tracknet_inputs = _prepare_tracknet_inputs(frames)
    yolo_inputs = _prepare_yolo_inputs(frames)

    latency_rows: list[dict[str, float | int | str]] = []
    memory_rows: list[dict[str, float | int | str]] = []
    summaries: list[ModelSummary] = []

    for name, model_path, model_inputs in (
        ("TrackNet", args.tracknet, tracknet_inputs),
        ("YOLO", args.yolo, yolo_inputs),
    ):
        model_latency, model_memory, summary = _run_model(
            name=name,
            model_path=model_path,
            inputs=model_inputs,
            runs=args.runs,
            warmup=args.warmup,
            provider=args.provider,
            process=process,
        )
        latency_rows.extend(model_latency)
        memory_rows.extend(model_memory)
        summaries.append(summary)
        print(
            f"{name}: latency mean={summary.mean_latency_ms:.2f} ms "
            f"std={summary.std_latency_ms:.2f} ms; "
            f"RSS mean={summary.mean_rss_mb:.1f} MB peak={summary.peak_rss_mb:.1f} MB"
        )

    _write_csv(args.output_dir / "latency.csv", latency_rows)
    _write_csv(args.output_dir / "memory.csv", memory_rows)

    summary_payload = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "provider": args.provider,
            "available_providers": ort.get_available_providers(),
            "video": str(args.video),
            "runs": args.runs,
            "frames": len(frames),
            "warmup": args.warmup,
        },
        "models": [asdict(summary) for summary in summaries],
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    _write_csv(args.output_dir / "summary.csv", [asdict(summary) for summary in summaries])

    _plot_latency(args.output_dir / "latency_over_time.png", latency_rows, summaries, len(frames))
    _plot_memory(args.output_dir / "memory_over_time.png", memory_rows, summaries, len(frames))
    _plot_memory_delta(args.output_dir / "memory_delta_over_time.png", memory_rows, summaries, len(frames))
    _plot_distribution(args.output_dir / "latency_distribution.png", latency_rows, summaries)

    print(f"Saved benchmark artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
