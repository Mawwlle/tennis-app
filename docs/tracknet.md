# TrackNet — детекция мяча

## Что такое TrackNet

Нейросеть для отслеживания маленьких быстрых мячей в видео. Собственная реализация UNet с depthwise separable свёртками.

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

Видео имеет исходное разрешение (например, 2914×1552), но TrackNet обучается на 640×360:

```python
TARGET_W = 640
TARGET_H = 360

frame = cv2.resize(frame, (TARGET_W, TARGET_H))
frame = frame.astype(np.float32) / 255.0  # нормализация к [0, 1]
```

Координаты аннотаций тоже масштабируются:

```python
cx_scaled = s.cx * 640 / s.orig_w
cy_scaled = s.cy * 360 / s.orig_h
```

---

## Архитектура модели (UNet + Depthwise Separable)

Модель — UNet с encoder-decoder и skip connections. Все свёртки — depthwise separable.

### Схема

```
Вход: (B, 9, 360, 640)
    │
    ▼
┌─────────────────────────────────┐ ЭНКОДЕР
│ enc1:  9 → 16 каналов           │  (B, 16, 360, 640)
│ pool + enc2: 16 → 32            │  (B, 32, 180, 320)
│ pool + enc3: 32 → 64            │  (B, 64, 90, 160)
│ pool + bottleneck: 64 → 128     │  (B, 128, 45, 80)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐ ДЕКОДЕР (со skip connections)
│ upsample + cat(e3) → dec3: 192→64  │  (B, 64, 90, 160)
│ upsample + cat(e2) → dec2: 96→32   │  (B, 32, 180, 320)
│ upsample + cat(e1) → dec1: 48→16   │  (B, 16, 360, 640)
└─────────────────────────────────┘
    │
    ▼
head Conv 1×1: 16 → 1
    │
    ▼
Выход: (B, 1, 360, 640) — logits тепловой карты
```

**Параметры: 63K** (вместо 15M у оригинального TrackNet).

### Depthwise Separable свёртки

Обычная Conv2d делает две вещи сразу: ищет паттерны **в пространстве** и смешивает каналы. Здесь разбиваем на два шага:

1. **Depthwise** — 3×3 свёртка отдельно по каждому каналу (`groups=in_ch`)
2. **Pointwise** — 1×1 свёртка для смешивания каналов

```python
def _dw_block(in_ch, out_ch) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),  # spatial
        nn.Conv2d(in_ch, out_ch, 1, bias=False),                           # channel mix
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
```

Даёт ~8× меньше параметров при схожем качестве.

### Skip connections

Encoder на каждом уровне сохраняет карты признаков `e1, e2, e3`. Decoder при апсемплинге склеивает их с текущей картой:

```python
x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
x = self.dec3(torch.cat([x, e3], dim=1))  # знаем ЧТО + детали enc3
```

Это позволяет декодеру точно локализовать мяч в пикселях — энкодер передаёт пространственные детали с каждого уровня.

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
        py, px = divmod(pred_map.argmax(), pred_map.shape[1])
        gy, gx = divmod(gt_map.argmax(), gt_map.shape[1])
        dist = sqrt((px - gx)² + (py - gy)²)
        if dist <= 5:   # допуск 5 пикселей в пространстве 640×360
            tp += 1
        else:
            fp += 1; fn += 1
```

**F1** = 2 × P × R / (P + R) — главная метрика. 5 пикселей допуска в 640×360 ≈ 0.5% ширины кадра.

---

## Обучение

### Разделение на train/val

Выборка делится не случайно, а **по временному порядку внутри каждого видео**.

```
normal_point/1.mov:
  80% → train: ранние кадры
  20% → val:   поздние кадры
```

Почему не случайно: соседние кадры очень похожи. Если перемешать, модель просто запомнит конкретные кадры вместо того чтобы научиться находить мяч.

### Предизвлечение кадров

Перед обучением `prepare_frames()` сохраняет все нужные кадры `[t-2, t-1, t]` в `dataset/frames/` как JPEG 640×360. Это решает проблему ненадёжного видеосикинга в DataLoader воркерах.

### Оптимизатор: Adadelta

```python
optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

Adadelta автоматически подстраивает learning rate. Если val_loss не улучшается 5 эпох → lr ÷ 2.

### Логи

Все метрики пишутся в консоль и в `weights/train.log`:

```
10:32:15  Epoch   1/100  train=0.0234  val=0.0198  P=0.421  R=0.380  F1=0.400
10:33:42  Epoch   5/100  train=0.0189  val=0.0176  P=0.512  R=0.498  F1=0.505
10:33:42    ✓ new best F1=0.505  saved → weights/tracknet_best.pt
```

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
    orig_w: int        # оригинальный размер видео
    orig_h: int
    visibility: int    # 1 = виден
```

### TrackNetDataset.__getitem__

```python
def __getitem__(self, idx):
    s = self.samples[idx]
    t = s.frame_idx

    # Тройка кадров из предизвлечённых JPEG-файлов
    frames = [_load_frame(s.video_path, max(0, t - 2)),
              _load_frame(s.video_path, max(0, t - 1)),
              _load_frame(s.video_path, t)]

    stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)

    cx_s = s.cx * TARGET_W / s.orig_w
    cy_s = s.cy * TARGET_H / s.orig_h
    heatmap = make_heatmap(cx_s, cy_s, TARGET_W, TARGET_H)

    return torch.from_numpy(stacked), torch.from_numpy(heatmap).unsqueeze(0)
    # shapes: (9, 360, 640)           (1, 360, 640)
```

Кадры читаются через `cv2.imread` из JPEG-файлов — быстро и надёжно.

---

## Инференс

```python
model = TrackNet()
model.load_state_dict(torch.load("weights/tracknet_best.pt", map_location=device))
model.eval()

# Подготовить три кадра
stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)  # (1, 9, 360, 640)

with torch.no_grad():
    logits = model(tensor)                   # (1, 1, 360, 640)
    heatmap = torch.sigmoid(logits)[0, 0]   # (360, 640), значения [0,1]

if heatmap.max() > 0.4:
    flat = heatmap.argmax().item()
    cy, cx = divmod(flat, 640)
    # Масштабировать обратно к оригинальным координатам
    cx_orig = cx * orig_w / 640
    cy_orig = cy * orig_h / 360
```