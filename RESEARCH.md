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
