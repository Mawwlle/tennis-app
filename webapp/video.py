from pathlib import Path

import cv2
from pydantic import BaseModel


class VideoInfo(BaseModel):
    model_config = {"from_attributes": True}

    video_id: str
    filename: str
    frame_count: int
    fps: float
    width: int
    height: int


class VideoEntry(BaseModel):
    video_id: str
    filename: str
    category: str


def _video_id_from_path(path: Path, dataset_dir: Path) -> str:
    return str(path.relative_to(dataset_dir))


def list_videos(dataset_dir: Path) -> list[VideoEntry]:
    entries: list[VideoEntry] = []
    for video_path in sorted(dataset_dir.rglob("*.mov")):
        video_id = _video_id_from_path(video_path, dataset_dir)
        entries.append(
            VideoEntry(
                video_id=video_id,
                filename=video_path.name,
                category=video_path.parent.name,
            )
        )
    return entries


def get_video_info(dataset_dir: Path, video_id: str) -> VideoInfo:
    video_path = dataset_dir / video_id
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    info = VideoInfo(
        video_id=video_id,
        filename=video_path.name,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def extract_frame(dataset_dir: Path, video_id: str, frame_idx: int) -> bytes:
    video_path = dataset_dir / video_id
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret: bool
    frame = None
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_id}")

    success: bool
    buf = None
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success or buf is None:
        raise RuntimeError(f"Failed to encode frame {frame_idx}")

    return buf.tobytes()
