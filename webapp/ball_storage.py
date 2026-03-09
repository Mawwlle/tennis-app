import json
from pathlib import Path

from model.schemas import BallAnnotation, BallAnnotationStore

BALL_ANNOTATIONS_PATH = Path("dataset/ball_annotations.json")


def load_ball_annotations(path: Path) -> BallAnnotationStore:
    if not path.exists():
        return BallAnnotationStore()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read ball annotations: {path}") from exc
    return BallAnnotationStore.model_validate(json.loads(raw))


def save_ball_annotations(store: BallAnnotationStore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write ball annotations: {path}") from exc


def add_ball_annotation(
    store: BallAnnotationStore, video_id: str, ann: BallAnnotation
) -> BallAnnotationStore:
    annotations = [a for a in store.videos.get(video_id, []) if a.frame_idx != ann.frame_idx]
    annotations.append(ann)
    annotations.sort(key=lambda a: a.frame_idx)
    return BallAnnotationStore(videos={**store.videos, video_id: annotations})


def remove_ball_annotation(
    store: BallAnnotationStore, video_id: str, frame_idx: int
) -> BallAnnotationStore:
    annotations = [a for a in store.videos.get(video_id, []) if a.frame_idx != frame_idx]
    return BallAnnotationStore(videos={**store.videos, video_id: annotations})
