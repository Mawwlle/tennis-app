# weights/

Веса моделей. Не коммитятся в git (добавлены в `.gitignore`).

## Файлы

| Файл | Размер | Описание |
|---|---|---|
| `tracknet_best.pt` | ~1 MB | TrackNet — лучшие веса по F1 на валидации. PyTorch формат. |
| `yolo_det.pt` | — | YOLO-детектор (мяч, стол, сетка, ракетки). Используется в webapp как fallback. |
| `tracknet.onnx` | ~1 MB | TrackNet, экспортированный в ONNX (opset 18). |
| `tracknet_opt.onnx` | ~918 KB | ONNX после упрощения графа через `onnxsim`. |
| `tracknet_int8.onnx` | ~264 KB | INT8-квантизованный ONNX. На ARM медленнее FP32 — не использовать. |

## Как получить

**Обучить TrackNet:**
```bash
uv run train
# сохраняет weights/tracknet_best.pt
```

**Экспортировать в ONNX:**
```bash
uv run python export_onnx.py
# сохраняет weights/tracknet.onnx, tracknet_opt.onnx, tracknet_int8.onnx
```

## Использование ONNX (без PyTorch)

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("weights/tracknet_opt.onnx", providers=["CPUExecutionProvider"])

# frames: 3 RGB-кадра, каждый (3, 360, 640), нормализованы в [0, 1]
# stack в (1, 9, 360, 640)
frames = np.zeros((1, 9, 360, 640), dtype=np.float32)
heatmap = sess.run(None, {"frames": frames})[0]  # (1, 1, 360, 640)

# Найти позицию мяча
flat = heatmap[0, 0].argmax()
cy, cx = divmod(int(flat), 640)
confidence = float(heatmap[0, 0].max())
```

## Производительность (CPU, Apple M-series)

| Формат | Время/кадр |
|---|---|
| PyTorch (.pt) | ~40 ms |
| ONNX raw | ~53 ms |
| ONNX opt | ~52 ms |
| ONNX int8 | ~109 ms ❌ (ARM не поддерживает эффективно) |