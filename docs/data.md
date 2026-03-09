# Данные и схемы

## Что где лежит

```
dataset/
  ball_annotations.json    — позиции мяча (cx, cy) по кадрам
  annotations.json         — игровые события (net/bounce/hit) по кадрам
  videos/
    normal_point/          — видео: обычный розыгрыш
    serve_into_net/        — видео: подача в сетку
    sasha_tichka/          — видео: ...
    soplya_setka/          — видео: ...

weights/
  tracknet_best.pt         — обученный TrackNet
  yolo_det.pt              — YOLO-детектор
```

---

## `ball_annotations.json`

Главный файл для обучения TrackNet.

### Структура

```json
{
  "videos": {
    "normal_point/1.mov": [
      {
        "frame_idx": 78,
        "cx": 526.54,
        "cy": 847.14,
        "visibility": 1,
        "source": "manual"
      },
      {
        "frame_idx": 79,
        "cx": 524.41,
        "cy": 851.39,
        "visibility": 1,
        "source": "yolo"
      }
    ],
    "serve_into_net/Untitled.mov": [...]
  }
}
```

### Поля

| Поле | Тип | Описание |
|---|---|---|
| `frame_idx` | int | Номер кадра в видео (0-based) |
| `cx` | float | Центр мяча по X в **оригинальных пикселях** видео |
| `cy` | float | Центр мяча по Y в **оригинальных пикселях** видео |
| `visibility` | int | `1` — мяч виден, `0` — мяча нет в кадре |
| `source` | string | `"manual"` — вручную, `"yolo"` — принята детекция YOLO |

### Важно про координаты

Координаты хранятся в пикселях **оригинального разрешения видео** (2914×1552). При обучении они масштабируются:

```
cx_for_training = cx * 640 / 2914
cy_for_training = cy * 360 / 1552
```

Текущее состояние:
- **725 аннотаций** по 3 видео
- Все `visibility=1` (мяч всегда виден)
- 418 аннотаций в `normal_point/1.mov` (кадры 78–517)
- 157 аннотаций в `serve_into_net/Screen Recording...` (кадры 0–206)
- 150 аннотаций в `serve_into_net/Untitled.mov` (кадры 8–176)

---

## `annotations.json`

Для будущего 3D CNN классификатора игровых событий. Пока мало заполнен.

### Структура

```json
{
  "videos": {
    "normal_point/1.mov": [
      {"frame_idx": 120, "label": "bounce"},
      {"frame_idx": 145, "label": "hit"}
    ]
  }
}
```

### Метки

| Метка | Событие |
|---|---|
| `"net"` | Мяч ударился о сетку |
| `"bounce"` | Мяч отскочил от стола |
| `"hit"` | Игрок ударил по мячу |

---

## `tennis-dataset/` — внешний датасет

Разметка стола, сетки и ракеток. **Мяча нет.**

### Формат меток (YOLO segmentation)

Каждый `.txt` файл соответствует одному PNG-кадру. Формат строки:

```
class_id  x1 y1  x2 y2  x3 y3  ...  xn yn
```

Все координаты нормализованы к `[0, 1]` относительно размера изображения.

Пример файла `frame_000498.txt`:
```
2 0.096 0.998 0.095 0.913 0.387 0.635 0.389 0.677
1 0.001 0.781 0.161 0.693 0.625 0.706 0.686 1.000 0.374 0.998 0.004 0.948
```

### Классы

| ID | Класс |
|---|---|
| `0` | ball (мяч) — **в этом датасете отсутствует** |
| `1` | table (стол) |
| `2` | grid (сетка) |
| `3` | racket (ракетка) |

### Конвертация в пиксели

Изображения 1280×720, поэтому:
```python
x_pixel = x_normalized * 1280
y_pixel = y_normalized * 720
```

---

## Pydantic схемы (`model/schemas.py`)

Pydantic — библиотека для валидации данных. Все структуры данных описаны как Pydantic-модели: при создании объекта автоматически проверяются типы.

### Схемы аннотаций

```python
class GameState(StrEnum):
    net = "net"
    bounce = "bounce"
    hit = "hit"

class Annotation(BaseModel):
    frame_idx: int
    label: GameState         # только "net", "bounce" или "hit"

class AnnotationStore(BaseModel):
    videos: dict[str, list[Annotation]] = {}
    # ключ = video_id, значение = список аннотаций
```

```python
class BallSource(StrEnum):
    yolo = "yolo"
    manual = "manual"

class BallAnnotation(BaseModel):
    frame_idx: int
    cx: float
    cy: float
    visibility: int          # 0 или 1
    source: BallSource       # только "yolo" или "manual"

class BallAnnotationStore(BaseModel):
    videos: dict[str, list[BallAnnotation]] = {}
```

### Схемы конфигурации моделей

```python
class ModelConfig(BaseModel):
    num_frames: int = 9        # длина последовательности для 3D CNN
    frame_height: int = 224
    frame_width: int = 224
    num_classes: int = 3       # net/bounce/hit
    hidden_dim: int = 256

class TrainingConfig(BaseModel):
    batch_size: int = 8
    learning_rate: float = 1e-3
    epochs: int = 30
    device: str = "cpu"
```

---

## Откуда берутся данные для обучения TrackNet

```
ball_annotations.json
    │
    ▼
train_tracknet.py: load_samples()
    │  Для каждой записи создаёт Sample:
    │  {video_path, frame_idx, cx, cy, orig_w, orig_h, visibility}
    ▼
tracknet/dataset.py: TrackNetDataset.__getitem__()
    │  Для кадра t:
    │  1. Читает кадры t-2, t-1, t из видеофайла
    │  2. Ресайзит каждый до 640×360
    │  3. Нормализует к [0,1]
    │  4. Стэкает в тензор (9, 360, 640)
    │  5. Масштабирует cx, cy
    │  6. Генерирует тепловую карту (1, 360, 640)
    ▼
train_loader / val_loader
    │  batch_size=4, 579 train / 146 val
    ▼
tracknet/train.py: train_epoch()
    │  forward → loss → backward → optimizer.step()
    ▼
tracknet_weights/tracknet_best.pt
```
