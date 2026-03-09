# TrackNet — детекция мяча

## Что такое TrackNet

TrackNet — это нейросеть специально для отслеживания маленьких быстрых мячей в видео (бадминтон, настольный теннис, теннис). Оригинальная работа: [github.com/yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet).

Ключевая идея: **вместо bbox предсказывать тепловую карту**.

```
Обычный подход (YOLO):
  Кадр → [нейросеть] → bbox [x, y, w, h] + confidence

TrackNet:
  3 кадра → [нейросеть] → тепловая карта (числа 0..1 в каждом пикселе)
                             ↓
                         найти максимум → координата мяча
```

---

## Тепловые карты (heatmaps)

### Что это

Тепловая карта — это изображение того же размера что и кадр, где яркость пикселя = вероятность того, что в этом месте находится мяч.

```
Пример тепловой карты (схематично):
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0.3 0.7 1.0 0.7 0.3 0
0 0 0 0 0.2 0.5 0.7 0.5 0.2 0
0 0 0 0 0 0 0 0 0 0

↑ Мяч находится в точке где значение = 1.0
```

### Зачем это лучше bbox

- **Устойчиво к смазу** — размытый мяч даёт широкий, но правильный пик на тепловой карте
- **Нет жёсткой границы** — модель выражает неуверенность через размытый пик
- **Работает с частично видимым мячом** — пик просто чуть слабее

### Гауссова функция

Тепловая карта генерируется как 2D-гауссиан (колокол) с центром в позиции мяча:

```
heatmap[y][x] = exp( -((x - cx)² + (y - cy)²) / (2 * σ²) )
```

Где:
- `cx`, `cy` — координаты мяча
- `σ = 5.0` — "ширина" колокола (5 пикселей в пространстве 640×360)

```python
# tracknet/heatmap.py
def make_heatmap(cx, cy, width, height, sigma=5.0):
    xs = np.arange(width)
    ys = np.arange(height)
    xg, yg = np.meshgrid(xs, ys)  # сетка координат
    heatmap = np.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2))
    return heatmap  # значения в [0, 1]
```

---

## Входные данные: 3 кадра

TrackNet видит не один кадр, а **три последовательных**: t−2, t−1, t.

```
Кадр t-2 (RGB, 3 канала)  ┐
Кадр t-1 (RGB, 3 канала)  ├─ склеить → 9-канальный тензор
Кадр t   (RGB, 3 канала)  ┘

Форма: (batch, 9, 360, 640)
```

Три кадра дают модели информацию о **траектории**: откуда летит мяч, с какой скоростью, в каком направлении.

### Масштаб кадров

Видео имеет разрешение 2914×1552, но TrackNet обучается на 640×360:

```python
TARGET_W = 640
TARGET_H = 360

frame = cv2.resize(frame, (TARGET_W, TARGET_H))
frame = frame.astype(np.float32) / 255.0  # нормализация к [0, 1]
```

Координаты аннотаций тоже масштабируются:

```python
cx_scaled = ann.cx * 640 / 2914   # ≈ ann.cx * 0.22
cy_scaled = ann.cy * 360 / 1552   # ≈ ann.cy * 0.23
```

---

## Архитектура модели (VGG-UNet)

Модель состоит из двух частей: энкодера (сжимает) и декодера (разворачивает обратно).

### Схема

```
Вход: (B, 9, 360, 640)
    │
    ▼
┌─────────────────────────────────┐ ЭНКОДЕР
│ VGG-блок 1: 9 → 64 каналов     │
│ MaxPool → 180×320               │
│                                 │
│ VGG-блок 2: 64 → 128 каналов   │
│ MaxPool → 90×160                │
│                                 │
│ VGG-блок 3: 128 → 256 каналов  │
│ MaxPool → 45×80                 │
│                                 │
│ Bottleneck: 256 → 512 каналов  │ ← самое "сжатое" представление
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐ ДЕКОДЕР
│ Upsample ×2 → 90×160           │
│ VGG-блок: 512 → 256 каналов    │
│                                 │
│ Upsample ×2 → 180×320          │
│ VGG-блок: 256 → 128 каналов    │
│                                 │
│ Upsample ×2 → 360×640          │
│ VGG-блок: 128 → 64 каналов     │
└─────────────────────────────────┘
    │
    ▼
Conv 1×1: 64 → 1 канал
    │
    ▼
Выход: (B, 1, 360, 640) — тепловая карта (logits)
```

### VGG-блок

```python
def _vgg_block(in_ch, out_ch, n):
    # n раз: Conv2D → ReLU → BatchNorm
    layers = []
    for i in range(n):
        ch_in = in_ch if i == 0 else out_ch
        layers += [Conv2d(ch_in, out_ch, 3, padding=1), ReLU(), BatchNorm2d(out_ch)]
    return Sequential(*layers)
```

**Conv2D** — свёрточный слой, извлекает паттерны
**ReLU** — активация, обнуляет отрицательные значения
**BatchNorm** — нормализует значения, стабилизирует обучение

### Downsampling и Upsampling

**MaxPool2d(2)** — сжимает карту признаков в 2 раза, берёт максимум из блока 2×2 пикселей.

**Bilinear interpolation** — растягивает карту признаков в 2 раза методом билинейной интерполяции:
```python
nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
```

Это стандартный способ восстановить пространственное разрешение в UNet-подобных архитектурах.

### Почему нет skip connections

В классическом UNet есть "мостики" (skip connections) между энкодером и декодером. TrackNet их не использует — это упрощает архитектуру и снижает риск переобучения на маленьком датасете.

---

## Функция потерь (Focal BCE)

```python
def focal_bce_loss(pred, target, gamma=2.0, pos_weight=50.0):
    # 1. Взвешенная бинарная кросс-энтропия
    bce = binary_cross_entropy_with_logits(pred, target, pos_weight=50)
    # 2. Focal-множитель — уменьшает вес "лёгких" примеров
    p_t = sigmoid(pred) * target + (1 - sigmoid(pred)) * (1 - target)
    focal = ((1 - p_t) ** gamma) * bce
    return focal.mean()
```

### Почему не обычная BCE

В тепловой карте 640×360 = 230 400 пикселей. Мяч занимает ≈ 50–100 пикселей. Остальные 230 000+ — фон (нули). Это сильный **дисбаланс классов**.

**pos_weight=50** — говорит модели, что ошибка на пикселе с мячом в 50 раз важнее чем ошибка на фоне.

**Focal (γ=2)** — дополнительно снижает вес "лёгких" примеров (очевидный фон), фокусируя обучение на сложных случаях.

---

## Метрики

Точность оценивается не по значениям пикселей, а по **расстоянию** между предсказанной и реальной позицией мяча.

```python
def _detection_metrics(pred_logits, target, threshold=0.5, dist_tol=5):
    for pred_map, gt_map in zip(preds, gts):
        # Найти предсказанную позицию — пиксель с максимальным значением
        py, px = divmod(pred_map.argmax(), pred_map.shape[1])

        # Найти реальную позицию
        gy, gx = divmod(gt_map.argmax(), gt_map.shape[1])

        # Расстояние между ними
        dist = sqrt((px - gx)² + (py - gy)²)

        if dist <= 5:   # допуск 5 пикселей в пространстве 640×360
            tp += 1     # правильно найден
        else:
            fp += 1; fn += 1  # нашли, но не там
```

**Precision** = TP / (TP + FP) — из всех предсказаний сколько правильных
**Recall** = TP / (TP + FN) — из всех реальных мячей сколько нашли
**F1** = 2 × P × R / (P + R) — среднее гармоническое, главная метрика

5 пикселей допуска в 640×360 = примерно 15 пикселей в оригинале 2914×1552 ≈ 0.5% ширины кадра.

---

## Обучение

### `train_tracknet.py` — точка входа

```python
device = torch.device("mps" if mps_available else "cuda" if cuda_available else "cpu")
# mps = Apple Silicon GPU, cuda = Nvidia GPU, cpu = процессор

samples = load_samples("annotations/ball_annotations.json", "dataset/")
train_loader, val_loader = build_loaders(samples, val_ratio=0.2, batch_size=4)
run_training(train_loader, val_loader, epochs=100, lr=1.0, ...)
```

### Разделение на train/val

Важный момент: выборка делится не случайно, а **по временному порядку внутри каждого видео**.

```
normal_point/1.mov: 418 аннотаций
  80% (334 кадра) → train: кадры с 78 по ~390
  20% ( 84 кадра) → val:   кадры с ~391 по 517
```

Почему не случайно: соседние кадры очень похожи. Если перемешать, модель просто запомнит конкретные кадры вместо того чтобы научиться находить мяч.

### Оптимизатор: Adadelta

```python
optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
```

Adadelta автоматически подстраивает learning rate для каждого параметра. lr=1.0 — стандартное значение для него (не значит что шаг огромный).

### Scheduler: ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

Если val_loss не улучшается 5 эпох подряд → уменьшить lr в 2 раза. Помогает доучиться после плато.

### Процесс

```
Каждую эпоху:
  for batch in train_loader:
    1. Прочитать 4 тройки кадров (batch_size=4)
    2. Прогнать через модель → получить 4 тепловые карты
    3. Вычислить focal BCE loss
    4. Backward → обновить веса

Каждые 5 эпох:
  Запустить на val_loader → вычислить F1
  Если F1 лучший → сохранить модель в tracknet_best.pt
  Если val_loss не улучшается 5 раз → уменьшить lr
```

### Логи

Все метрики пишутся в консоль и в файл `tracknet_weights/train.log`:

```
10:32:15  Epoch   1/100  train=0.0234  val=0.0198  P=0.421  R=0.380  F1=0.400
10:33:42  Epoch   5/100  train=0.0189  val=0.0176  P=0.512  R=0.498  F1=0.505
10:33:42    ✓ new best F1=0.505  saved → tracknet_weights/tracknet_best.pt
```

### Результат

Лучшая модель сохраняется в `tracknet_weights/tracknet_best.pt`. Это файл с весами нейросети (~50 МБ).

---

## `tracknet/dataset.py` — загрузка данных

### Sample — одна единица данных

```python
@dataclass
class Sample:
    video_path: Path   # путь к видео
    frame_idx: int     # индекс аннотированного кадра t
    cx: float          # координата мяча в оригинале
    cy: float
    orig_w: int        # оригинальный размер (2914)
    orig_h: int        # оригинальный размер (1552)
    visibility: int    # 1 = виден
```

### TrackNetDataset.__getitem__

```python
def __getitem__(self, idx):
    s = self.samples[idx]
    cap = cv2.VideoCapture(str(s.video_path))

    # Тройка кадров: t-2, t-1, t (с clamp: если t=0, то 0,0,0)
    idxs = [max(0, s.frame_idx - 2), max(0, s.frame_idx - 1), s.frame_idx]
    frames = [_read_frame(cap, i) for i in idxs]  # каждый: float32 (360, 640, 3)

    # Стэкнуть в 9 каналов: (3, H, W, 3) → transpose → (9, H, W)
    stacked = np.concatenate([f.transpose(2,0,1) for f in frames], axis=0)

    # Сгенерировать тепловую карту
    cx_s = s.cx * 640 / s.orig_w   # масштаб
    cy_s = s.cy * 360 / s.orig_h
    heatmap = make_heatmap(cx_s, cy_s, 640, 360)  # (360, 640)

    return torch.from_numpy(stacked), torch.from_numpy(heatmap).unsqueeze(0)
    # shapes: (9, 360, 640)           (1, 360, 640)
```

Каждый раз открывает видеофайл заново — медленно, зато не нужно держать в памяти все кадры.

---

## Инференс (как использовать обученную модель)

После обучения модель применяется так:

```python
model = TrackNet()
model.load_state_dict(torch.load("tracknet_weights/tracknet_best.pt"))
model.eval()

# Подготовить три кадра
frames = load_triplet(video, frame_idx)          # (9, 360, 640)
input_tensor = frames.unsqueeze(0)               # (1, 9, 360, 640)

with torch.no_grad():
    logits = model(input_tensor)                 # (1, 1, 360, 640)
    heatmap = torch.sigmoid(logits)[0, 0]        # (360, 640), значения [0,1]

# Найти позицию мяча
if heatmap.max() > 0.5:
    pos = heatmap.argmax()
    y, x = divmod(pos.item(), 640)
    # Масштабировать обратно к оригинальным координатам
    cx = x * orig_w / 640
    cy = y * orig_h / 360
```

Это логика, которую нужно реализовать в `apply_to_frame.py` после того как TrackNet обучен.
