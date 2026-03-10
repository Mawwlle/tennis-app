# YOLO модели

## Что такое YOLO

YOLO (You Only Look Once) — семейство моделей для детекции объектов.

На входе — изображение. На выходе — список объектов, каждый с:
- Классом (что это?)
- Bounding box (где это? — прямоугольник x, y, w, h)
- Confidence (насколько уверена модель, 0–1)

## Наши модели

Лежат в `weights/`.

### `weights/yolo_det.pt` — детекция

Возвращает прямоугольные bbox:

```
ball    → [x0, y0, x1, y1, confidence]
table   → [x0, y0, x1, y1, confidence]
grid    → [x0, y0, x1, y1, confidence]
racket  → [x0, y0, x1, y1, confidence]
```

### Классы

| ID | Класс | Цвет в визуализации |
|---|---|---|
| 0 | ball | cyan |
| 1 | table | green |
| 2 | grid (сетка) | orange |
| 3 | racket | purple |

## Как используются в проекте

### В веб-аннотаторе (`webapp/detector.py`)

Singleton — загружается один раз при первом запросе. Используется как fallback если TrackNet не загружен.

```python
def detect_ball(frame_bytes: bytes) -> BallDetectionResult | None:
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    result = _get_model()(frame, conf=0.15)[0]
    boxes = result.boxes[result.boxes.cls == 0]   # только мяч (class 0)
    if len(boxes) == 0:
        return None
    best = boxes.conf.argmax()
    x0, y0, x1, y1 = boxes.xyxy[best].tolist()
    return BallDetectionResult(cx=(x0+x1)/2, cy=(y0+y1)/2, ...)
```

Порог `conf=0.15` очень низкий — лучше лишний раз предложить неправильный вариант, чем пропустить мяч.

## Почему YOLO слабо находит мяч

1. **Маленький объект** — мяч занимает ~0.01% площади кадра
2. **Один кадр** — нет информации о движении
3. **Смаз** — при 60fps мяч летит быстро, остаётся смазанным

Именно поэтому основным детектором в проекте является TrackNet.