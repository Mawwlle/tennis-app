# Скрипты

Все скрипты запускаются через `uv run`:

```bash
uv run train    # обучение TrackNet
uv run infer    # применить модель к видео
uv run webapp   # запустить веб-аннотатор
```

---

## `train_tracknet.py` — обучение TrackNet

Читает аннотации, предизвлекает кадры, создаёт даталоадеры, запускает цикл обучения.

### Запуск

```bash
uv run train
# или: uv run python train_tracknet.py
```

### Параметры (менять прямо в файле)

```python
ANNOTATIONS_PATH = Path("dataset/ball_annotations.json")
DATASET_DIR      = Path("dataset/videos")
OUTPUT_DIR       = Path("weights")

EPOCHS     = 100
BATCH_SIZE = 4
LR         = 1.0    # learning rate для Adadelta
VAL_RATIO  = 0.2    # 20% данных — валидация
```

### Предизвлечение кадров

При первом запуске автоматически запускается `prepare_frames()` — все нужные кадры `[t-2, t-1, t]` сохраняются в `dataset/frames/` как JPEG 640×360. При повторных запусках пропускаются уже извлечённые файлы.

### Что сохраняется

```
weights/
  tracknet_best.pt   — веса с лучшим F1 на валидации
  train.log          — полный лог с временными метками
```

### Выбор устройства

Автоматически: MPS (Apple Silicon) → CUDA (Nvidia) → CPU.

---

## `infer_on_video.py` — применить TrackNet к видео

Запускает TrackNet и сохраняет видео с двумя панелями: слева — трекинг с траекторией, справа — тепловая карта.

### Запуск

```bash
uv run infer
# или: uv run python infer_on_video.py
# Результат: infer_result_new.mp4
```

### Параметры

```python
VIDEO_PATH     = Path("dataset/videos/normal_point/...")
WEIGHTS        = Path("weights/tracknet_best.pt")
OUTPUT         = Path("infer_result_new.mp4")

CONF_THRESHOLD = 0.4   # минимальная уверенность для детекции
TRAIL_WINDOW   = 9     # последние N детекций для трейла
INFER_STEP     = 3     # запускать TrackNet каждые N кадров
```

### Как работает

**Pass 1 (детекция):**

Проходит по видео один раз. TrackNet запускается каждые `INFER_STEP=3` кадра:

```python
if frame_idx % INFER_STEP == 0:
    cx, cy, conf, _ = predict(model, frames[-3:], device)
    if cx >= 0:
        detections[frame_idx] = (cx, cy)
```

Между кадрами где мяч найден — позиции заполняются сплайн-интерполяцией:

```python
spl_x = make_interp_spline(detected_frames, xs, k=3)
spl_y = make_interp_spline(detected_frames, ys, k=3)
```

**Pass 2 (рендер):**

Снова проходит по видео, рендерит для каждого кадра:

| Левая панель | Правая панель |
|---|---|
| Оригинальный кадр (640×360) | Тепловая карта INFERNO (640×360) |
| Жёлтый сплайн-трейл (последние 9 детекций) | — |
| Cyan кружок — обнаружен TrackNet | — |
| Синий кружок — интерполирован | — |

Итоговый размер видео: **1280×360**.

---

## `export_onnx.py` — экспорт в ONNX

Экспортирует обученный TrackNet в ONNX, упрощает граф через `onnxsim`, бенчмаркирует производительность.

### Запуск

```bash
uv run python export_onnx.py
```

### Что создаётся

```
weights/
  tracknet.onnx      — сырой ONNX (opset 18)
  tracknet_opt.onnx  — упрощённый через onnxsim
```

### Производительность (CPU, Apple M-series)

| Формат | Время/кадр |
|---|---|
| PyTorch (.pt) | ~40 ms |
| ONNX opt | ~52 ms |

Подробнее — в `weights/README.md`.

---

## `main.py` — веб-аннотатор

Запускает FastAPI-сервер с браузерным интерфейсом для разметки.

### Запуск

```bash
uv run webapp
# или: uv run python main.py
# Открыть: http://127.0.0.1:8000
```

Подробнее о веб-аннотаторе — в [webapp.md](webapp.md).