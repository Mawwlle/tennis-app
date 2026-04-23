# RESEARCH.md

Лог экспериментов и исследований по EventNet — классификатору игровых событий (hit / bounce / net).

---

## Эксперимент 1 — Координаты (cx, cy) + 1D Conv

**Дата**: 2026-04-03
**Статус**: ❌ провал (60% accuracy = предсказывает только "none")

### Что делали
- Вход: 9 нормализованных `(cx, cy)` → тензор `(B, 9, 2)`
- Модель: два слоя `Conv1d` → `GlobalAvgPool` → `Linear(32, 4)`
- Loss: `CrossEntropyLoss` без весов классов

### Результат
```
Total samples: 376
Label distribution: hit=50, bounce=66, net=6, none=254
Best val accuracy: 0.600 (epoch 1, не улучшилось за 200 эпох)
```

### Анализ
Две корневые причины провала:
1. **Дисбаланс классов**: 254/376 = 67.5% none → модель просто предсказывает none всегда
2. **Неправильная архитектура**: 1D Conv на сырых координатах не извлекает физически значимые признаки (скорость, ускорение)

---

## Эксперимент 2 — 9 Heatmap + 2D CNN + weighted loss

**Дата**: 2026-04-03
**Статус**: 🔄 в процессе обучения

### Что делали
- Вход: 9 Gaussian heatmap 64×36 → тензор `(B, 9, 36, 64)`
- Модель: `Conv2d(9, 16)` → `Conv2d(16, 32)` → `AdaptiveAvgPool` → `Linear(32, 4)`
- Loss: `CrossEntropyLoss` с inverse-frequency весами классов
- Веса: `[none=0.37, bounce=1.42, hit=1.88, net=15.6]`
- 300 эпох, Adam lr=1e-3, weight_decay=1e-3

### Гипотеза
Heatmap кодирует позицию пространственно → 2D Conv должен лучше извлекать паттерны движения, чем 1D Conv на координатах. Weighted loss должен починить предсказание minority классов.

### Ожидаемый результат
Нет результата пока — в процессе.

---

## Исследование — Что работает в литературе

**Дата**: 2026-04-03
**Источники**: TTNet (CVPR 2020), arxiv 2302.09657, arxiv 2104.09907, OpenTTGames dataset

### Ключевые находки

#### TTNet (CVPR 2020) — sota по задаче
- **97% event spotting accuracy**, >130 FPS при 1080p
- Вход: 9 последовательных кадров → global + local detection → event head
- **Smooth labels**: `target = sin(offset * π / 8)` в ±4 кадрах от события (вместо hard 0/1)
- **Multilabel sigmoid** (не softmax) — bounce и net могут активироваться одновременно
- Repo: https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch

#### 1D TCN на кинематических фичах (arxiv 2302.09657)
- **87% accuracy** на unseen players
- Вход: `[cx, cy, dx, dy, d²x, d²y, speed, angle]` × N кадров
- Архитектура: 1D TCN с dilation [1, 2, 4] outperforms LSTM и MLP
- Физический смысл фич:
  - **Bounce**: `dy` меняет знак (мяч летел вниз → летит вверх)
  - **Hit**: `dx` меняет знак + скачок скорости
  - **Net**: резкое падение speed, `cx` около центра кадра

#### Pose estimation для hit detection (arxiv 2104.09907)
- **99.37% accuracy** на 11 типах ударов используя 8 ключевых точек MediaPipe
- Hit по trajectory ненадёжен — мяч просто меняет направление без информации об источнике
- MediaPipe Holistic при 192×192 — ~20 FPS на мобильном CPU

#### Гомография стола → бесплатные геометрические фичи
- Один раз детектировать углы стола по цвету → homography matrix
- Добавить к вектору: `ball_over_table`, `ball_y_relative_to_table`, `ball_x_relative_to_net`
- Bounce физически возможен только при `ball_over_table=1`
- Net физически возможен только при `ball_x_relative_to_net ≈ 0`

#### OpenTTGames Dataset
- **4271 labeled event** (bounce/net) при 120fps — https://lab.osai.ai
- Бесплатный. Pre-train на нём → fine-tune на своих 376 сэмплах = +10–15% accuracy

### Рекомендуемый вектор признаков
```
На каждый кадр: [cx, cy, dx, dy, d²x, d²y, speed, angle]
Окно 9 кадров → 9 × 8 = 72 числа

vs текущий подход: 9 × 36 × 64 = 20 736 чисел (в 288× больше)
```

### Стратегии с малыми данными
- **T-SMOTE** (IJCAI 2022): oversampling для временных рядов с сохранением структуры — для 6 net сэмплов до ~30
- **Temporal augmentation**: jitter (шум к позиции), horizontal flip (`cx → W - cx` — стол симметричен), time stretch (stride 2)
- **Masked autoencoder pre-training**: маскировать 30–50% `(cx, cy)` в unlabeled видео, обучить TCN восстанавливать → fine-tune на labeled

### Счёт очков (FSM поверх классификатора)
```
WAIT → [hit] → RALLY
RALLY → [bounce своя сторона] → ошибка, очко сопернику
RALLY → [net] → проверить пересёк ли мяч → let или fault
RALLY → [bounce чужая сторона] → ждём ответный удар
RALLY → timeout (мяч вышел) → очко последнему бьющему
```
Debouncing: события в окне ±9 кадров объединяются, берём максимальный confidence.

---

## Эксперимент 3 — Кинематические фичи + 1D TCN

**Дата**: 2026-04-03
**Статус**: ✅ завершён

### Что делали
- Вход: `[cx, cy, dx, dy, d²x, d²y, speed, angle]` × 9 кадров → тензор `(B, 8, 9)`
- Модель: 3 `_TCNBlock` с dilation [1, 2, 4] + residual skip → `AdaptiveAvgPool1d` → `Linear(32, 4)`
- 7620 параметров, <50 KB ONNX
- Loss: `CrossEntropyLoss` с inverse-frequency весами
- 300 эпох, Adam lr=1e-3, weight_decay=1e-3

### Результат
```
Best val accuracy: 0.867 (epoch ~170–280)
Per-class: hit=0.73  bounce=0.88  net=0.00  none=0.93
Training time: ~40 секунд на MPS
```

### Анализ
- **Общая accuracy 0.867** — значительный прогресс по сравнению с 0.600 в эксперименте 1
- **Bounce (0.88) и none (0.93)** — учатся хорошо, кинематические фичи работают
- **Hit (0.73)** — умеренно, направление `dx` меняется не всегда однозначно
- **Net (0.00)** — полностью не обучается: всего 6 примеров (возможно, 0 попало в val)
- **Овerfitting**: val_loss растёт с ~1.0 до ~1.3 пока accuracy улучшается → модель запоминает train

### Почему лучше heatmap
- 2D CNN на heatmap сначала вынужден извлечь позицию из блоба — лишний шаг
- 72 числа vs 20 736 — в 288× меньше входное пространство
- Физика в явном виде: bounce → sign flip `dy`, hit → sign flip `dx` + скачок speed
- TCN с dilation покрывает весь 9-кадровый контекст (receptive field = 29)

---

## Эксперимент 4 — OpenTTGames pre-training (текущий)

**Дата**: 2026-04-03
**Статус**: ✅ завершён

### Что делали
- Скачали аннотации OpenTTGames (только JSON, без видео): 12 игр через `download_openttgames.py`
- Смаппировали: bounce→bounce, net→net, empty_event→пропускаем
- Объединили с собственными данными:
  - Own: hit=50, bounce=66, net=6, none=254 (376 total)
  - OpenTTGames: bounce=1771, net=1347 (3118 total)
  - **Combined: hit=50, bounce=1837, net=1353, none=254 (3494 total)**
- Class weights пересчитаны: hit=17.47, bounce=0.48, net=0.65, none=3.44
- 300 эпох, те же гиперпараметры

### Результат
```
Best val accuracy: 0.970 (epoch 225/300)
Per-class: hit=0.71  bounce=0.98  net=0.98  none=0.94
```

### Сравнение с экспериментом 3
| Метрика | Exp 3 (без OTT) | Exp 4 (с OTT) | Прирост |
|---------|-----------------|----------------|---------|
| accuracy | 0.867 | **0.970** | +10.3% |
| bounce | 0.88 | **0.98** | +10% |
| net | 0.00 | **0.98** | +98% |
| hit | 0.73 | **0.71** | −2% |
| none | 0.93 | **0.94** | +1% |

### Анализ
- **Net класс исправлен**: 0.00 → 0.98. OpenTTGames дал 1347 net-примеров вместо 6 — проблема решена полностью
- **Hit незначительно упал** (0.73 → 0.71): в combined датасете hit по-прежнему 50 примеров, но теперь соревнуется с 3494 общими; class weight высокий (17.47) но недостаточно
- **Главный bottleneck** теперь — hit-класс, нужно больше своих hit-аннотаций

---

## Эксперимент 5 — Убрать net-класс, добавить net geometry + player geometry фичи

**Дата**: 2026-04-23
**Статус**: ✅ завершён (обучение), ❌ inference плохой

### Что делали
- Убрали net-класс → 3 класса: hit / bounce / none
- Добавили 4 фичи расстояния до ракетки: dist_x/y_left, dist_x/y_right
- Убрали YOLO-seg из training pipeline (используем default NetGeometry)
- N_FEATURES: 10 → 14, num_classes: 4 → 3
- OpenTTGames: bounce только (net теперь пропускается)
- Smooth labels — нет (добавили позже)

### Результат обучения
```
Own samples: 372
OpenTTGames: 12 games, 1770 samples {'bounce': 1770}
Total: 2142 (hit=50, bounce=1838, none=254)
Best val accuracy: 0.993 (epoch 240)
Per-class: hit=0.90  bounce=0.99  none=1.00
Training time: ~2 мин на cuda
```

### Проблема
**Hit детектируется там где не должен** — inference не работает.
- val accuracy 99.3% но false positive rate высокий на реальном видео
- Корень: запускаем EventNet на *каждом кадре* all_positions (~5000+ кадров)
- При 1% false positive rate → 50 ложных hit на 5000 кадров
- Kinematic features noisy: интерполяция сглаживает артефакты детекции, модель учится на "чистых" данных, inference получает "грязные"

---

## Эксперимент 6 — HeatmapEventNet (TTNet approach)

**Дата**: 2026-04-23
**Статус**: 🔄 в процессе

### Идея (из TTNet, CVPRW 2020)
TTNet не использует скалярные (cx, cy) — он берёт **feature maps из детектора** как вход для event head. Ключевые идеи:
1. **Spatial context**: сеть видит *где в кадре* мяч — низ кадра = стол = bounce, верх = игрок = hit
2. **Sin smooth labeling**: `weight = sin((R - |d| + 1) * π / (2*(R+1)))` для d ∈ {-R…R} вокруг каждого события → в 7× больше hit samples (50 → 350)
3. **BCEWithLogitsLoss** (sigmoid, не softmax) — каждый класс независим
4. **Heatmap dropout 50%** при обучении → имитирует INFER_STEP=3 (inference видит только каждый 3-й кадр)
5. **Confidence threshold** вместо argmax + peak picking → не классифицируем каждый кадр подряд

### Архитектура HeatmapEventNet
```
Input: (B, 15, 36, 64)  — window of downscaled ball detection heatmaps

Shared spatial encoder (per frame):
  Conv2d(1 → 16, 3×3) → BN → ReLU
  Conv2d(16 → 24, 3×3, stride=2) → BN → ReLU  # 18×32
  Conv2d(24 → 32, 3×3, stride=2) → BN → ReLU  # 9×16
  AdaptiveAvgPool2d(1) → Flatten                 # → 32-dim per frame

Temporal TCN: (B, 32, 15) → 3 dilated blocks → (B, 64, 15)
Head: AdaptiveAvgPool1d → Flatten → Dropout(0.3) → Linear(64, 3)

Output: (B, 3) raw logits for BCEWithLogitsLoss
Params: ~100K    Inference: < 3ms on CPU
```

### Ключевые изменения в pipeline
- **Training**: синтетические Gaussian blobs из аннотаций (cx, cy) → 36×64 heatmaps
- **Inference** (`score.py`): Pass 1 сохраняет downscaled TrackNet heatmaps в `hm_cache` → detect_hits использует реальные heatmaps
- `openttgames.py`: переписан на `build_openttgames_heatmap_samples` с smooth labeling
- `train_eventnet.py`: убраны player_geometry и net_geometry pre-computation
- `eventnet/train.py`: `_weighted_bce` с per-sample weights, CosineAnnealingLR

### Файлы
- `eventnet/heatmap.py` — новый: HeatmapEventSample, HeatmapEventDataset, build_heatmap_samples, smooth labeling utils
- `eventnet/model.py` — добавлены HeatmapEventNet, HeatmapEventNetConfig (TCNEventNet сохранён как legacy)
- `eventnet/train.py` — переписан на BCEWithLogitsLoss + weighted loss
- `eventnet/openttgames.py` — переписан на heatmap samples

---

## Эксперимент 7 — Rule-based scoring без EventNet

**Дата**: 2026-04-23
**Статус**: 🔄 в процессе

### Решение
- Оставляем `TrackNet + interpolation + net geometry`
- Полностью убираем `EventNet` из `score.py`
- **Hit** детектируем по локальным экстремумам `x(t)`:
  - локальный минимум = удар слева
  - локальный максимум = удар справа
- **Bounce** детектируем по локальным максимумам `y(t)` после сглаживания траектории
- **Net** детектируем как неудачный перелёт после `hit`:
  - был удар
  - не было валидного bounce на чужой стороне
  - траектория либо закончилась у сетки, либо не пересекла её
- Поверх событий строим FSM для счёта:
  - `hit` → ждём bounce на стороне соперника
  - `bounce` → ждём ответный `hit`
  - второй `bounce` на той же стороне = очко сопернику
  - `net` или потеря мяча до валидного bounce = очко сопернику

### Почему
- Трекинг уже хороший, а `eventnet` даёт слишком много ложных срабатываний
- Для score overlay важнее стабильная физика ралли, чем frame-wise классификация каждого кадра
- Эвристики по траектории проще отлаживать на edge device и легче интерпретировать

---

## Эксперимент 8 — Hit pseudo-labels из OpenTTGames траекторий

**Дата**: 2026-04-23
**Статус**: 🔄 запланирован

### Проблема
OpenTTGames не размечает хиты (только bounce/net/empty_event).
Из-за этого hit обучается только на 50 собственных сэмплах → hit accuracy = 0.71.

### Решение
Генерировать hit псевдо-разметку из `ball_markup.json` через детекцию смены знака dx:
- Хит = локальный экстремум cx: `dx[f] * dx[f+1] < 0` с `speed > _HIT_MIN_DX`
- Фильтрация: cx вне net-зоны (0.38–0.62) и `_HIT_MIN_EVENT_GAP = 10` от размеченных событий
- Smooth labeling (±3 кадра) применяется так же, как к размеченным bounce

### Изменения
- `eventnet/openttgames.py`: добавлена `_detect_hit_frames()` + интеграция в `build_openttgames_heatmap_samples`
- Параметры: `_HIT_MIN_DX=0.008` (≈15px при 1920), `_HIT_NET_ZONE=(0.38, 0.62)`, `_HIT_MAX_FRAME_GAP=4`

### Ожидаемый результат
- Hit: 0.71 → ~0.90+ (если OpenTTGames ball_markup.json покрывает весь ролик)
- Общая accuracy: ~97% → ~99%+

---

## Следующие шаги (приоритет)

| # | Что | Ожидаемый прирост | Усилие |
|---|---|---|---|
| 1 | 1D TCN + кинематические фичи (dx, dy, d²x, d²y, speed) | +15–20% | низкое |
| 2 | Weighted CE + undersample none до 2:1 | модель перестаёт предсказывать только none | низкое |
| 3 | Smooth labels (TTNet-style) | +3–5%, стабильнее | низкое |
| 4 | Гомография стола + 3 геометрические фичи | +5–8% на bounce/net | среднее |
| 5 | Скачать OpenTTGames, pre-train | +10–15% | среднее |
| 6 | MediaPipe wrist velocity для hit | +5–10% на hit | высокое |

**Реалистичный потолок без OpenTTGames**: ~75–80%
**С OpenTTGames pre-training**: ~85–90%
**Главное ограничение**: 6 сэмплов net — класс нестабилен пока нет больше данных
