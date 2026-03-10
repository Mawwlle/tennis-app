# Данные и схемы

## Что где лежит

```
dataset/
  ball_annotations.json    — позиции мяча (cx, cy) по кадрам
  annotations.json         — игровые события (net/bounce/hit) по кадрам
  videos/
    normal_point/          — видео: обычный розыгрыш
    serve_into_net/        — видео: подача в сетку
    sasha_tichka/
    soplya_setka/
  frames/                  — предизвлечённые JPEG-кадры для обучения
    normal_point/
      video_name/
        000078.jpg         — кадр с номером 78
        ...

weights/
  tracknet_best.pt         — обученный TrackNet
  tracknet_opt.onnx        — ONNX после упрощения
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
        "source": "tracknet"
      }
    ]
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
| `source` | string | `"manual"`, `"yolo"` или `"tracknet"` |

### Важно про координаты

Координаты хранятся в пикселях **оригинального разрешения видео**. При обучении они масштабируются:

```python
cx_for_training = cx * 640 / orig_w
cy_for_training = cy * 360 / orig_h
```

---

## `annotations.json`

Для 3D CNN классификатора игровых событий.

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

## Pydantic схемы (`model/schemas.py`)

### Схемы аннотаций

```python
class GameState(StrEnum):
    net = "net"
    bounce = "bounce"
    hit = "hit"

class Annotation(BaseModel):
    frame_idx: int
    label: GameState

class AnnotationStore(BaseModel):
    videos: dict[str, list[Annotation]] = {}
```

```python
class BallSource(StrEnum):
    yolo = "yolo"
    manual = "manual"
    tracknet = "tracknet"

class BallAnnotation(BaseModel):
    frame_idx: int
    cx: float
    cy: float
    visibility: int          # 0 или 1
    source: BallSource

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
tracknet/dataset.py: prepare_frames()
    │  Предизвлекает кадры [t-2, t-1, t] из видео в dataset/frames/
    │  Пропускает уже существующие файлы
    ▼
tracknet/dataset.py: TrackNetDataset.__getitem__()
    │  Читает кадры t-2, t-1, t через cv2.imread из JPEG
    │  Стэкает в тензор (9, 360, 640)
    │  Масштабирует cx, cy
    │  Генерирует тепловую карту (1, 360, 640)
    ▼
train_loader / val_loader
    │  batch_size=4, num_workers=4, pin_memory=True
    ▼
tracknet/train.py: train_epoch()
    │  forward → focal_bce_loss → backward → optimizer.step()
    ▼
weights/tracknet_best.pt
```