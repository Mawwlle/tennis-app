# Tennis App

ML-пайплайн для детекции мяча и классификации игровых событий в настольном теннисе + веб-приложение для разметки видео.

---

## Структура проекта

```
dataset/
  videos/                     — исходные видео (.mov) по категориям
    normal_point/
    ...
  frames/                     — предизвлечённые кадры для обучения (генерируются автоматически)
  ball_annotations.json       — разметка мяча (cx, cy, visibility, source)
  annotations.json            — разметка событий (hit / bounce / net)
model/                        — Pydantic-схемы (GameState, BallAnnotation, ...)
tracknet/                     — детектор мяча
  model.py                    — UNet с depthwise separable свёртками (63K параметров)
  dataset.py                  — PyTorch Dataset с предизвлечением кадров
  train.py                    — тренировочный цикл (focal BCE, Adadelta)
  heatmap.py                  — генерация Gaussian heatmap
eventnet/                     — классификатор игровых событий
  model.py                    — HeatmapEventNet: 2D CNN, вход 9 heatmap → 4 класса
  dataset.py                  — датасет из окон ball_annotations + events
  train.py                    — тренировочный цикл (cross-entropy с весами классов)
webapp/                       — FastAPI + одностраничный аннотатор
  static/index.html           — SPA с двумя режимами разметки
  detector.py                 — YOLO-детектор мяча
  tracknet_detector.py        — TrackNet-детектор мяча
  ball_storage.py             — хранилище ball-аннотаций
weights/
  tracknet_best.pt            — лучшие веса TrackNet по F1 (~1 MB)
  eventnet_best.pt            — лучшие веса HeatmapEventNet по accuracy
  tracknet_opt.onnx           — ONNX после упрощения через onnxsim (~918 KB)
  yolo_det.pt                 — YOLO детекция мяча
train_tracknet.py             — точка входа: обучение TrackNet
train_eventnet.py             — точка входа: обучение HeatmapEventNet
infer_on_video.py             — инференс на видео, side-by-side вывод + метки событий
export_onnx.py                — экспорт TrackNet в ONNX + бенчмарк
main.py                       — запуск веб-приложения
```

---

## Быстрый старт

```bash
uv sync
uv run webapp      # открыть http://127.0.0.1:8000
```

---

## Обучение

### TrackNet (детекция мяча)

```bash
uv run train
# веса → weights/tracknet_best.pt
```

Аннотации читаются из `dataset/ball_annotations.json`. При первом запуске кадры автоматически предизвлекаются в `dataset/frames/` как JPEG 640×360.

### HeatmapEventNet (классификация событий)

```bash
uv run train-events
# веса → weights/eventnet_best.pt
```

Требует размеченных событий в `dataset/annotations.json` и позиций мяча в `dataset/ball_annotations.json`.

### Unified Calibration Segmentation

```bash
uv run train-seg
# веса → weights/calibration_seg_best.pt
# график метрик → weights/calibration_seg_metrics.png
```

Этот пайплайн автоматически:
- докачивает OpenTTGames training videos и segmentation masks
- собирает единый YOLO-seg датасет из локальных `net/table` аннотаций и OpenTTGames `person/table`
- обучает один segmentation model для `net / table / person`

Важно: OpenTTGames не размечает `net`, поэтому класс сетки по-прежнему приходит из локальной разметки.

---

## Инференс на видео

```bash
uv run infer
# результат: infer_result_new.mp4
```

Вывод — два экрана рядом:
- **Слева**: трекинг мяча + сплайн-трейл + метка события (hit / bounce / net)
- **Справа**: тепловая карта (INFERNO colormap)

TrackNet запускается каждые 3 кадра (`INFER_STEP=3`), промежутки заполняются сплайн-интерполяцией. Если `weights/eventnet_best.pt` не существует — событийная классификация пропускается.

---

## Экспорт в ONNX

```bash
uv run python export_onnx.py
# weights/tracknet.onnx, tracknet_opt.onnx
```

Производительность на CPU (Apple M-series):

| Формат | Время/кадр |
|---|---|
| PyTorch (.pt) | ~40 ms |
| ONNX opt | ~52 ms |

---

## Веб-аннотатор

### Режим Ball (разметка мяча)

При загрузке кадра автоматически запускается TrackNet-детектор (если включён).

| Клавиша / действие | Результат |
|---|---|
| `Space` | принять детекцию (source=tracknet или yolo) |
| клик на кадр | поставить мяч вручную (source=manual) |
| `V` | пометить кадр как «мяч не виден» (visibility=0) |
| `Esc` | сбросить pending-детекцию |
| `←` / `→` | навигация по кадрам |

**Auto-markup full video** — кнопка запускает TrackNet на всём видео за один проход.

### Режим Events (игровые события)

| Клавиша | Действие |
|---|---|
| `←` / `→` | ±1 кадр |
| `Shift+←` / `Shift+→` | ±10 кадров |
| `N` | net |
| `B` | bounce |
| `H` | hit |
| клик по таймлайну | переход к кадру |

---

## Архитектура

### TrackNet

UNet с depthwise separable свёртками.

- **Вход**: 3 последовательных кадра → `(B, 9, 360, 640)`
- **Выход**: heatmap logits `(B, 1, 360, 640)`
- **Параметры**: 63K
- **Каналы**: 9 → 16 → 32 → 64 → 128 → 64 → 32 → 16 → 1

Почему лучше YOLO для мяча:

| | YOLO | TrackNet |
|---|---|---|
| Контекст | 1 кадр | 3 кадра (видит движение) |
| Малый размер мяча | плохо | хорошо |
| Смаз при быстром движении | теряет | обрабатывает |

### HeatmapEventNet

2D CNN-классификатор игровых событий по траектории мяча.

- **Вход**: 9 последовательных heatmap → `(B, 9, 36, 64)`
- **Выход**: `(B, 4)` — логиты классов hit / bounce / net / none
- **Параметры**: ~15K

Каждая heatmap — Gaussian-блоб в позиции мяча (σ=3 px при разрешении 64×36). 2D свёртки по 9 каналам обучаются распознавать пространственный паттерн движения:
- **hit**: резкое изменение направления
- **bounce**: отскок от стола — смена знака вертикальной скорости
- **net**: остановка / хаотичное движение

Дисбаланс классов компенсируется весами обратной частоты в `cross_entropy`.
