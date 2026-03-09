from enum import StrEnum

from pydantic import BaseModel


class GameState(StrEnum):
    net = "net"
    bounce = "bounce"
    hit = "hit"


class BallSource(StrEnum):
    yolo = "yolo"
    manual = "manual"
    tracknet = "tracknet"


class ModelConfig(BaseModel):
    num_frames: int = 9
    frame_height: int = 224
    frame_width: int = 224
    num_classes: int = 3
    hidden_dim: int = 256


class TrainingConfig(BaseModel):
    batch_size: int = 8
    learning_rate: float = 1e-3
    epochs: int = 30
    device: str = "cpu"


class Annotation(BaseModel):
    frame_idx: int
    label: GameState


class VideoAnnotations(BaseModel):
    video_path: str
    annotations: list[Annotation]


class AnnotationStore(BaseModel):
    videos: dict[str, list[Annotation]] = {}


class BallAnnotation(BaseModel):
    frame_idx: int
    cx: float
    cy: float
    visibility: int = 1  # 1 = visible, 0 = not in frame
    source: BallSource


class BallAnnotationStore(BaseModel):
    videos: dict[str, list[BallAnnotation]] = {}
