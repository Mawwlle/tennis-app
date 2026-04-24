# Table Tennis ML Pipeline

ML-пайплайн для трекинга мяча, детекции событий и автоматического счёта в настольном теннисе.

---

## Высокоуровневая архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          score.py  —  полный пайплайн                       │
│                                                                             │
│  ┌──────────────┐   ┌──────────────────────────────────────────────────┐   │
│  │   Pass 0     │   │                    Pass 1                         │   │
│  │              │   │                                                   │   │
│  │  YOLOv8n-seg │──▶│  TrackNet  ──▶  spline interp  ──▶  hm_cache    │   │
│  │  (каждые     │   │  (каждые 4 кадра)                                │   │
│  │  100 кадров) │   │                                                   │   │
│  └──────────────┘   └───────────────────────┬──────────────────────────┘   │
│         │                                    │                              │
│         │ net geometry                       │ ball positions + heatmaps    │
│         ▼                                    ▼                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                            Pass 2                                    │   │
│  │                                                                      │   │
│  │  HeatmapEventNet  ──▶  bounce detection (sigmoid peaks)             │   │
│  │  trajectory heuristics  ──▶  hit / net / miss detection             │   │
│  └───────────────────────────────────┬──────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Pass 3  ─  RallyStateMachine  ─▶  score timeline                   │   │
│  └───────────────────────────────────┬──────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Pass 4  ─  render annotated MP4  +  TTS audio                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Компоненты

### 1. TrackNet — детектор мяча

```
Входные данные
──────────────
  frame[t-2]  ─┐
  frame[t-1]  ─┼──▶  (B, 9, 360, 640)  ← 3 RGB-кадра склеены по каналам
  frame[t]    ─┘

Архитектура UNet
────────────────

  Энкодер
  ───────
  (B,  9, 360, 640)  ──▶  enc1: DW×2  ──▶  (B,  16, 360, 640)  ─── skip1
                               MaxPool 2×
  (B,  16, 180, 320)  ──▶  enc2: DW×2  ──▶  (B,  32, 180, 320)  ─── skip2
                               MaxPool 2×
  (B,  32,  90, 160)  ──▶  enc3: DW×2  ──▶  (B,  64,  90, 160)  ─── skip3
                               MaxPool 2×
  (B,  64,  45,  80)  ──▶  bottleneck: DW×2  ──▶  (B, 128, 45, 80)

  Декодер
  ───────
  (B, 128, 45, 80)   ──▶  upsample 2×  ──▶  cat(skip3)  ──▶  dec3: DW×2  ──▶  (B,  64,  90, 160)
  (B,  64, 90, 160)  ──▶  upsample 2×  ──▶  cat(skip2)  ──▶  dec2: DW×2  ──▶  (B,  32, 180, 320)
  (B,  32, 180, 320) ──▶  upsample 2×  ──▶  cat(skip1)  ──▶  dec1: DW×2  ──▶  (B,  16, 360, 640)
  (B,  16, 360, 640) ──▶  Conv2d(1×1)  ──▶  (B, 1, 360, 640)  ← logits heatmap

  DW = depthwise separable block: Conv2d(groups=in_ch, 3×3) + Conv2d(1×1) + BN + ReLU
       ~8× меньше параметров, чем стандартная свёртка

Выходные данные
───────────────
  (B, 1, 360, 640)  — логиты;  sigmoid  →  вероятность наличия мяча в каждом пикселе

Параметры: 63 K
```

**Обучение:**

| Параметр | Значение |
|---|---|
| Loss | Focal BCE (`pos_weight=50`, `gamma=2`) |
| Optimizer | Adadelta (`lr=1.0`) |
| Scheduler | ReduceLROnPlateau |
| Вход | 3 кадра → tensor (B, 9, 360, 640) |
| Разметка | Gaussian blob σ=10 px в позиции мяча |

Почему лучше YOLO для мяча:

| | YOLO | TrackNet |
|---|---|---|
| Контекст | 1 кадр | 3 кадра — видит движение |
| Малый мяч (~5 px) | плохо | хорошо |
| Смаз при быстром движении | теряет | обрабатывает |
| Параметры | 3M+ | **63K** |

---

### 2. HeatmapEventNet — детектор событий

Основан на подходе из TTNet (Voeikov et al., CVPRW 2020).

```
Входные данные
──────────────
  Окно из 15 последовательных TrackNet-heatmaps:
  каждая heatmap: Gaussian blob σ=2 px в позиции мяча, разрешение 36×64
  тензор: (B, 15, 36, 64)

Архитектура
───────────

  Пространственный энкодер (shared weights по всем 15 кадрам)
  ────────────────────────────────────────────────────────────
  Разворачиваем: (B, 15, 36, 64) → (B×15, 1, 36, 64)

  Conv2d(1→16, 3×3)  + BN + ReLU                →  (B×15, 16, 36, 64)
  Conv2d(16→24, 3×3, stride=2)  + BN + ReLU     →  (B×15, 24, 18, 32)
  Conv2d(24→32, 3×3, stride=2)  + BN + ReLU     →  (B×15, 32,  9, 16)
  AdaptiveAvgPool2d(1)  + Flatten                →  (B×15, 32)

  Собираем обратно: (B, 15, 32) → transpose → (B, 32, 15)

  Временной TCN (dilated convolutions)
  ─────────────────────────────────────
  _TCNBlock(32→64, dilation=1)   →  (B, 64, 15)   ← short-range context
  _TCNBlock(64→64, dilation=2)   →  (B, 64, 15)   ← medium-range
  _TCNBlock(64→64, dilation=4)   →  (B, 64, 15)   ← long-range

  _TCNBlock: Conv1d(dilation) + BN + ReLU + Dropout(0.2)  +  residual skip

  Голова классификатора
  ──────────────────────
  AdaptiveAvgPool1d(1) + Flatten  →  (B, 64)
  Dropout(0.3)
  Linear(64 → 2)                  →  (B, 2)  logits

Выходные данные
───────────────
  (B, 2)  — логиты: [bounce_score, none_score]
  sigmoid(logits[0]) > 0.50  →  bounce event

Параметры: ~44 K
```

**Подход TTNet: smooth temporal labeling**

Вместо одного сэмпла на каждое событие создаём `2×R+1` сэмплов с убывающими весами:

```
  Event at frame F, radius R=3:

  F-3  ──▶  weight = sin(1·π/8) ≈ 0.38
  F-2  ──▶  weight = sin(2·π/8) ≈ 0.71
  F-1  ──▶  weight = sin(3·π/8) ≈ 0.92
  F    ──▶  weight = sin(4·π/8) = 1.00  ← центр события
  F+1  ──▶  weight = 0.92
  F+2  ──▶  weight = 0.71
  F+3  ──▶  weight = 0.38
```

Это даёт ~7× больше сэмплов и устойчивость к неточности разметки (±1–2 кадра).

**Heatmap dropout 50%:** во время обучения случайно зануляем половину кадров окна — имитирует `INFER_STEP=4` (TrackNet запускается не каждый кадр).

**Обучение:**

| Параметр | Значение |
|---|---|
| Loss | BCEWithLogitsLoss (independent sigmoid per class) |
| Веса сэмплов | smooth TTNet-weight встроен в loss |
| Данные | OpenTTGames (12 игр) + собственная разметка |
| Neg ratio | 3:1 (none:bounce) |
| Классы | bounce(0), none(1) |

---

### 3. YOLOv8n-seg — сегментация стола и игроков

```
  Вход: кадр 640×360

  Детектируемые классы:
    0 — table   (стол целиком)
    1 — person  (игроки)

  Выход: маски сегментации

  Использование в пайплайне:
  ───────────────────────────
  Pass 0:  однократный запуск → CalibrationData
             table mask  ──▶  _build_net_mask_from_table()  ──▶  net geometry
             person masks ──▶  player bounding boxes

  Pass 1:  re-run каждые 100 кадров (SEG_INTERVAL)
           игроки движутся → обновляем player masks

  Веса: weights/seg_best.pt
  Данные для обучения: OpenTTGames dataset (PNG masks, R=table, G=person)
```

---

### 4. Инференс-пайплайн (`score.py`)

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Pass 0  —  Calibration                                                    │
│                                                                            │
│  YOLOv8n-seg(frame[0])  ──▶  table mask                                   │
│  _build_net_mask_from_table()  ──▶  net_x, net_y_top, net_y_bot           │
│  CalibrationData(net, table, players)                                      │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Pass 1  —  Ball tracking                                                  │
│                                                                            │
│  for frame in video:                                                       │
│      if frame % INFER_STEP == 0:                                           │
│          heatmap = TrackNet(frame[t-2], frame[t-1], frame[t])             │
│          hm_cache[frame_idx] = heatmap                                     │
│          pos = argmax(heatmap) if max > CONF_THRESHOLD else None           │
│      if frame % SEG_INTERVAL == 0:                                         │
│          calib = _try_update_calibration(seg_model, frame)  # refresh     │
│                                                                            │
│  spline interpolation  ──▶  smooth positions for all frames               │
│  savgol_filter  ──▶  trajectory smoothing                                  │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Pass 2  —  Event detection                                                │
│                                                                            │
│  EventNet path (if weights/eventnet_best.pt exists):                      │
│    sliding window over hm_cache  ──▶  sigmoid(logits[0])  ──▶  peaks     │
│    peaks > EVENTNET_THRESHOLD  ──▶  bounce events                         │
│                                                                            │
│  Trajectory heuristics (always active):                                   │
│    dy sign reversal + Y in table zone  ──▶  bounce                        │
│    dx sign reversal near net_x  ──▶  net                                   │
│    large dx + away from net  ──▶  hit                                      │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Pass 3  —  Scoring (RallyStateMachine)                                    │
│                                                                            │
│  idle  ──hit──▶  await_bounce  ──bounce──▶  await_return                  │
│                                ──net/miss──▶  point  ──▶  idle            │
│                 ──timeout──▶  idle                                          │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Pass 4  —  Render                                                         │
│                                                                            │
│  annotated MP4  +  TTS score audio (macOS say / gtts)                     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Структура проекта

```
dataset/
  videos/                     — исходные видео (.mov)
  frames/                     — предизвлечённые кадры (генерируются автоматически)
  openttgames/                — OpenTTGames dataset (game_1…test_4)
  ball_annotations.json       — разметка мяча (cx, cy, visibility, source)
  annotations.json            — разметка событий (bounce / hit)
model/
  schemas.py                  — Pydantic-схемы (BallAnnotation, AnnotationStore, …)
tracknet/
  model.py                    — UNet с depthwise separable свёртками (63K)
  dataset.py                  — PyTorch Dataset с предизвлечением кадров
  train.py                    — цикл обучения (focal BCE, Adadelta)
  heatmap.py                  — генерация Gaussian heatmap
eventnet/
  model.py                    — HeatmapEventNet: spatial CNN + dilated TCN (~44K)
  dataset.py                  — датасет из аннотированных событий
  heatmap.py                  — heatmap generation + HeatmapEventDataset
  openttgames.py              — парсер OpenTTGames dataset
  train.py                    — цикл обучения (BCE + smooth labels)
  features.py                 — кинематические фичи (legacy)
  segmentation.py             — загрузка YOLOv8n-seg
webapp/
  static/index.html           — SPA-аннотатор
  routes.py                   — API: детекция мяча + хранение аннотаций
  detector.py                 — YOLO-детектор мяча
  tracknet_detector.py        — TrackNet-детектор мяча
weights/
  tracknet_best.pt            — TrackNet (~1 MB)
  eventnet_best.pt            — HeatmapEventNet
  seg_best.pt                 — YOLOv8n-seg (table + person)
  yolov8n.pt                  — YOLOv8n для веб-аннотатора
train_tracknet.py             — точка входа: обучение TrackNet
train_eventnet.py             — точка входа: обучение HeatmapEventNet
train_seg.py                  — точка входа: обучение seg-модели
score.py                      — полный пайплайн (score_result.mp4)
infer_on_video.py             — side-by-side инференс (tracking + heatmap)
main.py                       — веб-аннотатор
```

---

## Быстрый старт

```bash
uv sync

# Веб-аннотатор (разметка мяча и событий)
uv run webapp               # http://127.0.0.1:8000

# Обучение
uv run train                # TrackNet → weights/tracknet_best.pt
uv run train-events         # HeatmapEventNet → weights/eventnet_best.pt
uv run seg                  # YOLOv8n-seg → weights/seg_best.pt

# Инференс
uv run infer                # side-by-side MP4: трекинг + heatmap
uv run python score.py      # полный пайплайн с счётом → score_result.mp4

# Экспорт
uv run python export_onnx.py  # ONNX + бенчмарк
```

---

## Производительность

| Модель | Параметры | Время/кадр (CPU, M-series) |
|---|---|---|
| TrackNet (PyTorch) | 63K | ~40 ms |
| TrackNet (ONNX opt) | 63K | ~52 ms |
| HeatmapEventNet | ~44K | < 3 ms / окно |
| YOLOv8n-seg | 3.4M | ~25 ms |

TrackNet запускается каждые `INFER_STEP=4` кадра; промежутки заполняются сплайн-интерполяцией. YOLOv8n-seg запускается один раз + каждые 100 кадров.

---

## Данные

### OpenTTGames dataset

12 игр (game_1…test_4), каждая содержит:
- `ball_markup.json` — позиции мяча `{"frame": {"x": px, "y": px}}`
- `events_markup.json` — события `{"frame": "bounce|net|empty_event"}`
- `*.png` маски сегментации (B-канал = стол, G-канал = игроки)

Используется для обучения EventNet (4271 bounce-событие) и seg-модели.

### Собственная разметка

- `dataset/ball_annotations.json` — ручная/TrackNet/YOLO разметка мяча
- `dataset/annotations.json` — разметка bounce/hit событий

Редактируется через веб-аннотатор.

---

## Веб-аннотатор

### Режим Ball

| Действие | Результат |
|---|---|
| `Space` | принять детекцию (source=tracknet/yolo) |
| клик | поставить мяч вручную |
| `V` | пометить как невидимый (visibility=0) |
| `←` / `→` | навигация по кадрам |
| Auto-markup | TrackNet по всему видео за один проход |

### Режим Events

| Клавиша | Действие |
|---|---|
| `B` | bounce |
| `H` | hit |
| `N` | net |
| `Shift+←` / `Shift+→` | ±10 кадров |
| клик по таймлайну | переход к кадру |

---

## Ссылки

- [TTNet: Real-time temporal and spatial video analysis of table tennis](https://arxiv.org/abs/2004.09927) — Voeikov et al., CVPRW 2020
- [OpenTTGames dataset](https://lab.osai.ai/) — 12 игр с разметкой мяча и событий
- `RESEARCH.md` — эксперименты, метрики, приоритизированный план улучшений
