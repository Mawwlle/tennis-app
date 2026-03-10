# Веб-аннотатор

## Что это

Одностраничное браузерное приложение для разметки видео. Открываешь `http://127.0.0.1:8000`, выбираешь видео, листаешь кадры и размечаешь мяч. Всё сохраняется автоматически в JSON.

## Запуск

```bash
uv sync          # установить зависимости
uv run webapp    # запустить сервер
```

## Архитектура

```
Браузер (index.html)
    ↕ HTTP запросы
FastAPI (webapp/routes.py)
    ↕
  ├── webapp/video.py             — читает видеофайлы через OpenCV
  ├── webapp/storage.py           — читает/пишет annotations.json
  ├── webapp/ball_storage.py      — читает/пишет ball_annotations.json
  ├── webapp/detector.py          — YOLO singleton (fallback для мяча)
  └── webapp/tracknet_detector.py — TrackNet singleton (основной детектор мяча)
```

Весь фронтенд — один файл `webapp/static/index.html`. Никаких фреймворков, чистый JS.

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

### `webapp/routes.py` — API эндпоинты

#### Видео

| Метод | URL | Что делает |
|---|---|---|
| `GET` | `/api/videos` | Список всех `.mov` файлов из `dataset/videos/` |
| `GET` | `/api/videos/{id}/info` | Метаданные: кол-во кадров, fps, разрешение |
| `GET` | `/api/videos/{id}/frame/{idx}` | Один кадр в формате JPEG (байты) |

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
| `GET` | `/api/videos/{id}/frame/{idx}/detect_ball` | Запустить TrackNet/YOLO, вернуть позицию мяча |
| `POST` | `/api/videos/{id}/detect_all` | Запустить TrackNet на всём видео, заполнить все кадры |

### `webapp/tracknet_detector.py` — TrackNet singleton

Основной детектор мяча. Загружается из `weights/tracknet_best.pt` при первом запросе.

```python
def detect_frame(video_path, frame_idx, orig_w, orig_h) -> TrackNetResult | None:
    # Берёт тройку кадров [t-2, t-1, t], прогоняет через TrackNet
    # Возвращает (cx, cy, confidence) в координатах оригинального видео

def detect_all_frames_batch(video_path, orig_w, orig_h) -> list[tuple[int, TrackNetResult]]:
    # Открывает видео один раз, прогоняет TrackNet на каждом кадре
    # Возвращает список (frame_idx, result) для всех кадров с conf > порога
```

### `webapp/detector.py` — YOLO singleton

Fallback-детектор из `weights/yolo_det.pt`. Используется если TrackNet не загружен.

```python
def detect_ball(frame_bytes: bytes) -> BallDetectionResult | None:
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    result = _get_model()(frame, conf=0.15)[0]
    boxes = result.boxes[result.boxes.cls == 0]   # только мяч (class 0)
    ...
```

### `webapp/video.py` — работа с видео

```python
def extract_frame(dataset_dir: Path, video_id: str, frame_idx: int) -> bytes:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
```

Возвращает JPEG-байты — браузер отображает их напрямую через `<img src="/api/videos/.../frame/42">`.

### `webapp/storage.py` и `webapp/ball_storage.py` — хранилище

Читают/пишут JSON-файл. Функции иммутабельные (возвращают новый объект, не мутируют старый).

```python
def add_ball_annotation(store, video_id, ann) -> BallAnnotationStore:
    annotations = [a for a in store.videos.get(video_id, []) if a.frame_idx != ann.frame_idx]
    annotations.append(ann)
    annotations.sort(key=lambda a: a.frame_idx)
    return BallAnnotationStore(videos={**store.videos, video_id: annotations})
```

---

## Фронтенд (`webapp/static/index.html`)

Весь интерфейс — один HTML-файл. Никаких фреймворков, чистый JavaScript.

### Режимы работы

#### Режим Events (события)

Размечаешь моменты игры:
- `N` — сетка (net)
- `B` — отскок (bounce)
- `H` — удар (hit)

#### Режим Ball (разметка мяча)

Размечаешь позицию мяча на каждом кадре:

```
Загрузился кадр
    ↓
Если включён Model-assisted detection:
    → запрос к /detect_ball (TrackNet)
    → показать пунктирный круг с уверенностью

  TrackNet нашёл мяч?
  ├── ДА  → показать результат
  │         Space = принять → сохранить как source="tracknet"
  │         Esc = отклонить → можно кликнуть вручную
  └── НЕТ → "not detected — click to place"
            Клик на кадр → сохранить как source="manual"
            V = мяч не виден → сохранить visibility=0
```

**Model-assisted detection** — переключатель в UI, отключает автодетекцию при навигации по кадрам. Полезно при ручной разметке или медленном железе.

**Auto-markup full video** — кнопка запускает TrackNet на всём видео за один проход (`POST /detect_all`). Потом можно отредактировать результат вручную.

### Canvas-оверлей

Поверх `<img>` с кадром лежит прозрачный `<canvas>` того же размера. На canvas рисуются аннотации.

**Пересчёт координат:** видео имеет оригинальное разрешение, но в браузере отображается меньше.

```javascript
function origToCanvas(cx, cy) {
    const rect = frameImg.getBoundingClientRect();
    return [cx * rect.width / frameOrigW, cy * rect.height / frameOrigH];
}
function canvasToOrig(cx, cy) {
    const rect = frameImg.getBoundingClientRect();
    return [cx * frameOrigW / rect.width, cy * frameOrigH / rect.height];
}
```

В JSON-файл сохраняются координаты в **оригинальных пикселях видео**. При обучении они масштабируются под TrackNet (640×360).

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

### `dataset/annotations.json` — события

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

### `dataset/ball_annotations.json` — мяч

```json
{
  "videos": {
    "normal_point/1.mov": [
      {
        "frame_idx": 78,
        "cx": 526.54,
        "cy": 847.14,
        "visibility": 1,
        "source": "tracknet"
      }
    ]
  }
}
```

`cx` и `cy` — координаты **центра** мяча в пикселях оригинального видео. При обучении масштабируются к 640×360.

`source` — одно из: `"manual"`, `"yolo"`, `"tracknet"`.