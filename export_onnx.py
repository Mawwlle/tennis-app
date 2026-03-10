"""Export TrackNet to ONNX and optimize for CPU inference."""

from pathlib import Path

import onnx
import onnxruntime as ort
import onnxsim
import torch
import numpy as np
from onnxruntime.quantization import QuantType, quantize_dynamic

from tracknet.model import TrackNet

WEIGHTS     = Path("weights/tracknet_best.pt")
OUT_RAW     = Path("weights/tracknet.onnx")
OUT_OPT     = Path("weights/tracknet_opt.onnx")
OUT_INT8    = Path("weights/tracknet_int8.onnx")

TARGET_W = 640
TARGET_H = 360


def export(weights: Path, out: Path) -> None:
    device = torch.device("cpu")
    model = TrackNet()
    model.load_state_dict(torch.load(str(weights), map_location=device, weights_only=True))
    model.eval()

    dummy = torch.zeros(1, 9, TARGET_H, TARGET_W)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["frames"],
        output_names=["heatmap"],
        dynamic_axes={"frames": {0: "batch"}, "heatmap": {0: "batch"}},
        opset_version=18,
    )
    print(f"Exported → {out}")


def simplify(src: Path, dst: Path) -> None:
    model = onnx.load(str(src))
    model_sim, ok = onnxsim.simplify(model)
    if not ok:
        print("WARNING: simplification did not fully succeed, saving as-is")
        model_sim = model
    onnx.save(model_sim, str(dst))
    print(f"Simplified → {dst}")


def quantize(src: Path, dst: Path) -> None:
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    print(f"Quantized (INT8) → {dst}")


def benchmark(path: Path, label: str) -> None:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 9, TARGET_H, TARGET_W), dtype=np.float32)

    # Warmup
    for _ in range(3):
        sess.run(None, {"frames": dummy})

    import time
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {"frames": dummy})
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {label}: {ms:.1f} ms/frame")


if __name__ == "__main__":
    export(WEIGHTS, OUT_RAW)
    simplify(OUT_RAW, OUT_OPT)
    quantize(OUT_OPT, OUT_INT8)

    print("\nBenchmark (CPU):")
    benchmark(OUT_RAW,  "raw  ")
    benchmark(OUT_OPT,  "opt  ")
    benchmark(OUT_INT8, "int8 ")

    sizes = {p.name: f"{p.stat().st_size / 1024:.0f} KB" for p in [OUT_RAW, OUT_OPT, OUT_INT8]}
    print("\nFile sizes:")
    for name, size in sizes.items():
        print(f"  {name}: {size}")
