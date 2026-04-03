# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research

Эксперименты, что пробовалось и результаты — в `RESEARCH.md`. Там же приоритизированный план улучшений EventNet и ссылки на релевантные статьи (TTNet, OpenTTGames dataset и др.).

## Commands

```bash
uv sync                          # Install dependencies
uv run webapp                    # Start web server (http://127.0.0.1:8000)
uv run train                     # Train TrackNet model
uv run infer                     # Run inference on video
uv run python export_onnx.py     # Export to ONNX + quantize + benchmark
```

No test runner or linter is configured.

## Architecture

**Tennis ball detection pipeline** with three components:

### 1. Web Annotator (`webapp/`, `main.py`)
FastAPI server (`create_app()` → `webapp/app.py`) with a single-page HTML frontend. Serves video frames as JPEG via API and persists annotations to JSON files:
- `dataset/ball_annotations.json` — ball positions per video/frame
- `dataset/annotations.json` — game events (net/bounce/hit)

Detection endpoints try **TrackNet first**, fall back to **YOLO** (`webapp/routes.py`).

Both model detectors are **module-level singletons** (lazy-loaded on first call) — acceptable pattern for expensive GPU resources (`webapp/detector.py`, `webapp/tracknet_detector.py`).

### 2. TrackNet (`tracknet/`)
Lightweight UNet (63K params) for ball detection:
- **Input**: 3 stacked RGB frames → tensor `(B, 9, 360, 640)`
- **Output**: heatmap logits `(B, 1, 360, 640)` — Gaussian peak at ball position
- **Loss**: Focal BCE with `pos_weight=50.0`, `gamma=2.0`
- **Optimizer**: Adadelta (`lr=1.0`), scheduler: ReduceLROnPlateau
- **Architecture**: Depthwise separable convolutions (`model.py`) for parameter efficiency

Training is orchestrated in `train_tracknet.py` → `tracknet/train.py`. Frames are pre-extracted to `dataset/frames/` as JPEG before training.

Weights live in `weights/tracknet_best.pt`. ONNX variants in `weights/tracknet*.onnx`.

### 3. EventNet (`eventnet/`)
Классификатор игровых событий поверх TrackNet:
- **Input**: 9 Gaussian heatmap позиций мяча → `(B, 9, 36, 64)`
- **Output**: 4 класса — hit / bounce / net / none
- **Loss**: CrossEntropy с inverse-frequency весами (none:bounce:hit:net ≈ 1:4:5:62)
- Обучается через `uv run train-events` → `weights/eventnet_best.pt`
- При инференсе `weights/eventnet_best.pt` опциональны — если нет, классификация пропускается

Текущий статус и следующие шаги — в `RESEARCH.md`. Основная проблема: сильный дисбаланс классов (6 net-сэмплов), лучшая стратегия — переход на кинематические фичи + 1D TCN.

### 4. Inference Pipeline (`infer_on_video.py`)
Two-pass video processing:
1. Run TrackNet every `INFER_STEP=3` frames, interpolate gaps with scipy spline
2. Render side-by-side MP4: left = tracking trail, right = INFERNO heatmap

### Data Contracts (`model/schemas.py`)
All shared types are Pydantic models: `BallAnnotation`, `BallAnnotationStore`, `BallSource` (StrEnum: yolo/manual/tracknet), `GameState` (StrEnum: net/bounce/hit).

Storage functions are **pure and immutable** — `add_ball_annotation()` returns a new store, never mutates in place.

### Device Detection
TrackNet detector auto-selects: MPS (Apple Silicon) → CUDA → CPU (`webapp/tracknet_detector.py`).
