# YOLO модели

## Что такое YOLO

YOLO (You Only Look Once) — семейство моделей для детекции объектов. "Only Once" означает что модель обрабатывает изображение один раз (в отличие от старых двухэтапных подходов).

На входе — изображение. На выходе — список объектов, каждый с:
- Классом (что это?)
- Bounding box (где это? — прямоугольник x, y, w, h)
- Confidence (насколько уверена модель, 0–1)

## Наши модели

Обе лежат в корне проекта (`yolo_det.pt`, `yolo_seg.pt`), обе обучены на данных настольного тенниса.

### `yolo_det.pt` — детекция

Возвращает прямоугольные bbox:

```
ball    → [x0, y0, x1, y1, confidence]
table   → [x0, y0, x1, y1, confidence]
grid    → [x0, y0, x1, y1, confidence]
racket  → [x0, y0, x1, y1, confidence]
```

### `yolo_seg.pt` — сегментация

Дополнительно к bbox возвращает полигон маски — точный контур объекта:

```
table → bbox + mask polygon [(x1,y1), (x2,y2), ...]
```

Сегментация точнее bbox, но медленнее.

### Классы

| ID | Класс | Цвет в визуализации |
|---|---|---|
| 0 | ball | cyan |
| 1 | table | green |
| 2 | grid (сетка) | orange |
| 3 | racket | purple |

## Как используются в проекте

### В веб-аннотаторе (`webapp/detector.py`)

Singleton — загружается один раз при первом запросе:

```python
_model = YOLO("yolo_det.pt")

def detect_ball(frame_bytes: bytes) -> BallDetectionResult | None:
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    result = _model(frame, conf=0.15)[0]              # conf=0.15 — низкий порог
    boxes = result.boxes[result.boxes.cls == 0]       # только мяч (class 0)
    if len(boxes) == 0:
        return None
    best = boxes.conf.argmax()
    x0, y0, x1, y1 = boxes.xyxy[best].tolist()
    return BallDetectionResult(cx=(x0+x1)/2, cy=(y0+y1)/2, conf=boxes.conf[best], ...)
```

### В `apply_to_frame.py`

Детекция на каждом кадре видео через `stream=True` — не загружает всё видео в память:

```python
for result in model(str(video_path), stream=True, conf=0.15):
    # result — результат для одного кадра
    ball = find_best_ball(result)
```

### В `yolo_detect.py`

Обе модели на семплированных кадрах, результат side-by-side.

## Почему YOLO слабо находит мяч

1. **Маленький объект** — мяч занимает ~0.01% площади кадра (2914×1552 → ~50×50 пикселей)
2. **Один кадр** — нет информации о движении
3. **Смаз** — при 60fps мяч летит быстро, остаётся смазанным

Именно поэтому мы переходим на TrackNet.

## Confidence threshold

В разных местах используются разные пороги:

| Место | Порог | Почему |
|---|---|---|
| `webapp/detector.py` | 0.15 | Лучше предложить неточное чем пропустить |
| `apply_to_frame.py` | 0.15 | То же самое |
| `yolo_detect.py` | 0.25 | Для визуализации нужно качество |

При низком пороге YOLO находит больше, но чаще ошибается. Пользователь сам решает принять или нет.
