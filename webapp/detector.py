from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from ultralytics import YOLO

BALL_CLASS_ID = 0
CONF_THRESHOLD = 0.15
WEIGHTS_PATH = Path("weights/yolo_det.pt")

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


class BallDetectionResult(BaseModel):
    cx: float
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float


def detect_ball(frame_bytes: bytes) -> BallDetectionResult | None:
    buf = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame: NDArray[np.uint8] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    model = _get_model()
    result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    boxes = result.boxes[result.boxes.cls == BALL_CLASS_ID]
    if len(boxes) == 0:
        return None

    best = int(boxes.conf.argmax())
    x0, y0, x1, y1 = boxes.xyxy[best].tolist()
    return BallDetectionResult(
        cx=(x0 + x1) / 2,
        cy=(y0 + y1) / 2,
        x0=x0, y0=y0, x1=x1, y1=y1,
        conf=float(boxes.conf[best]),
    )
