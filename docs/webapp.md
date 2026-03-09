# Веб-аннотатор

## Что это

Одностраничное браузерное приложение для разметки видео. Открываешь `http://127.0.0.1:8000`, выбираешь видео, листаешь кадры и размечаешь мяч. Всё сохраняется автоматически в JSON.

## Запуск

```bash
uv sync          # установить зависимости
python main.py   # запустить сервер
```

## Архитектура

```
Браузер (index.html)
    ↕ HTTP запросы
FastAPI (webapp/routes.py)
    ↕
  ├── webapp/video.py        — читает видеофайлы через OpenCV
  ├── webapp/storage.py      — читает/пишет annotations.json
  ├── webapp/ball_storage.py — читает/пишет ball_annotations.json
  └── webapp/detector.py     — запускает YOLO на кадре
```

Весь фронтенд — один файл `webapp/static/index.html` (~260 строк JavaScript). Никаких фреймворков, чистый JS.

---

## Бэкенд (FastAPI)

### `webapp/app.py` — фабрика приложения

```python
def create_app() -> FastAPI:
    app = FastAPI(title="Tennis Annotation Tool")
    app.include_router(router)         # подключить API
    app.mount("/", StaticFiles(...))   # отдавать index.html
    return app
```

Всё приложение создаётся в одной функции. Статические файлы монтируются последними (иначе они перехватят все запросы, включая `/api/...`).

### `webapp/routes.py` — API эндпоинты

#### Видео

| Метод | URL | Что делает |
|---|---|---|
| `GET` | `/api/videos` | Список всех `.mov` файлов из `dataset/` |
| `GET` | `/api/videos/{id}/info` | Метаданные: кол-во кадров, fps, разрешение |
| `GET` | `/api/videos/{id}/frame/{idx}` | Один кадр в формате JPEG (байты) |

Кадры читаются по запросу — не кешируются, не загружаются заранее. Каждый запрос открывает видеофайл, перемотает к нужному кадру, закрывает.

#### Событийные аннотации (net/bounce/hit)

| Метод | URL | Что делает |
|---|---|---|
| `GET` | `/api/videos/{id}/annotations` | Список аннотаций для видео |
| `POST` | `/api/videos/{id}/annotations` | Добавить аннотацию `{frame_idx, label}` |
| `DELETE` | `/api/videos/{id}/annotations/{frame_idx}` | Удалить аннотацию |

#### Ball-аннотации

| Метод | URL | Что делает |
|---|---|---|
| `GET` | `/api/videos/{id}/ball` | Список ball-аннотаций |
| `POST` | `/api/videos/{id}/ball` | Добавить `{frame_idx, cx, cy, visibility, source}` |
| `DELETE` | `/api/videos/{id}/ball/{frame_idx}` | Удалить аннотацию |
| `GET` | `/api/videos/{id}/frame/{idx}/detect_ball` | Запустить YOLO, вернуть bbox мяча |

### `webapp/video.py` — работа с видео

```python
def extract_frame(dataset_dir: Path, video_id: str, frame_idx: int) -> bytes:
    # Открыть видео через OpenCV
    cap = cv2.VideoCapture(str(video_path))
    # Перемотать к нужному кадру
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    # Прочитать кадр
    ret, frame = cap.read()
    # Сжать в JPEG (качество 85%)
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # Вернуть байты
    return buf.tobytes()
```

Возвращает JPEG-байты — браузер отображает их напрямую через `<img src="/api/videos/.../frame/42">`.

### `webapp/storage.py` и `webapp/ball_storage.py` — хранилище

Оба работают одинаково: читают/пишут JSON-файл, функции иммутабельные (возвращают новый объект, не мутируют старый).

```python
def add_ball_annotation(store, video_id, ann) -> BallAnnotationStore:
    # Убрать старую аннотацию для этого кадра (если была)
    annotations = [a for a in store.videos.get(video_id, []) if a.frame_idx != ann.frame_idx]
    # Добавить новую
    annotations.append(ann)
    # Отсортировать по номеру кадра
    annotations.sort(key=lambda a: a.frame_idx)
    # Вернуть новый store (не изменяя старый)
    return BallAnnotationStore(videos={**store.videos, video_id: annotations})
```

Каждый POST-запрос: читаем файл → модифицируем → пишем файл. Просто и надёжно.

### `webapp/detector.py` — YOLO singleton

```python
_model: YOLO | None = None  # глобальная переменная

def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolo_det.pt")  # загружается один раз при первом вызове
    return _model
```

YOLO-модель тяжёлая (~50–100 МБ, долго грузится). Поэтому она загружается один раз при первом запросе и живёт в памяти до конца работы сервера. Это называется «singleton» — один экземпляр на всё приложение.

```python
def detect_ball(frame_bytes: bytes) -> BallDetectionResult | None:
    # Декодировать JPEG обратно в массив пикселей
    buf = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    # Запустить YOLO
    result = _get_model()(frame, conf=0.15, verbose=False)[0]
    # Оставить только класс 0 (мяч)
    boxes = result.boxes[result.boxes.cls == BALL_CLASS_ID]
    if len(boxes) == 0:
        return None
    # Взять бокс с наибольшей уверенностью
    best = int(boxes.conf.argmax())
    x0, y0, x1, y1 = boxes.xyxy[best].tolist()
    return BallDetectionResult(cx=(x0+x1)/2, cy=(y0+y1)/2, ...)
```

Порог `conf=0.15` очень низкий — лучше лишний раз предложить неправильный вариант, чем пропустить мяч. Пользователь сам решает, принять или нет.

---

## Фронтенд (`webapp/static/index.html`)

Весь интерфейс — один HTML-файл. Никаких фреймворков, чистый JavaScript.

### Режимы работы

#### Режим Events (события)

Размечаешь моменты игры:
- `N` — сетка (net)
- `B` — отскок (bounce)
- `H` — удар (hit)

Используется для 3D CNN классификатора игровых состояний.

#### Режим Ball (разметка мяча)

Размечаешь позицию мяча на каждом кадре:

```
Загрузился кадр
    ↓
Автоматически отправляется запрос к /detect_ball
    ↓
  YOLO нашёл мяч?
  ├── ДА  → показать жёлтый bbox с уверенностью
  │         Space = принять → сохранить как source="yolo"
  │         Esc = отклонить → можно кликнуть вручную
  └── НЕТ → "not detected — click to place"
            Клик на кадр → сохранить как source="manual"
            V = мяч не виден → сохранить visibility=0
```

### Canvas-оверлей

Поверх `<img>` с кадром лежит прозрачный `<canvas>` того же размера. На canvas рисуются аннотации (кружки, bbox).

**Проблема координат:** Видео имеет разрешение 2914×1552, но в браузере отображается гораздо меньше (например, 1280×720). Нужно пересчитывать координаты.

```javascript
function origToCanvas(cx, cy) {
    const rect = frameImg.getBoundingClientRect(); // размер img в браузере
    return [
        cx * rect.width  / frameOrigW,  // frameOrigW = 2914
        cy * rect.height / frameOrigH,  // frameOrigH = 1552
    ];
}

function canvasToOrig(cx, cy) {
    const rect = frameImg.getBoundingClientRect();
    return [
        cx * frameOrigW / rect.width,
        cy * frameOrigH / rect.height,
    ];
}
```

В JSON-файл сохраняются координаты в **оригинальных пикселях видео** (2914×1552). При обучении они масштабируются под TrackNet (640×360).

### Два таймлайна

```
events  [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
         красные=net, голубые=bounce, золотые=hit

ball    [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
         зелёные=visible, серые=hidden
```

Клик по таймлайну → переход к этому кадру.

---

## Форматы файлов аннотаций

### `annotations/annotations.json` — события

```json
{
  "videos": {
    "normal_point/1.mov": [
      {"frame_idx": 120, "label": "bounce"},
      {"frame_idx": 145, "label": "hit"},
      {"frame_idx": 201, "label": "net"}
    ]
  }
}
```

### `annotations/ball_annotations.json` — мяч

```json
{
  "videos": {
    "normal_point/1.mov": [
      {
        "frame_idx": 78,
        "cx": 526.54,   // центр мяча по X в пикселях оригинального видео
        "cy": 847.14,   // центр мяча по Y
        "visibility": 1, // 1 = виден, 0 = не виден
        "source": "manual" // "manual" или "yolo"
      }
    ]
  }
}
```

`cx` и `cy` — координаты **центра** мяча в пикселях оригинального видео (2914×1552). Для обучения TrackNet они масштабируются к 640×360.
