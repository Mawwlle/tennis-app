import json
from pathlib import Path

from model.schemas import Annotation, AnnotationStore


def load_annotations(path: Path) -> AnnotationStore:
    if not path.exists():
        return AnnotationStore()

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read annotations file: {path}") from exc

    data = json.loads(raw)
    return AnnotationStore.model_validate(data)


def save_annotations(store: AnnotationStore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write annotations file: {path}") from exc


def add_annotation(store: AnnotationStore, video_id: str, annotation: Annotation) -> AnnotationStore:
    annotations = list(store.videos.get(video_id, []))
    # Replace existing annotation at same frame
    annotations = [a for a in annotations if a.frame_idx != annotation.frame_idx]
    annotations.append(annotation)
    annotations.sort(key=lambda a: a.frame_idx)
    updated_videos = {**store.videos, video_id: annotations}
    return AnnotationStore(videos=updated_videos)


def remove_annotation(store: AnnotationStore, video_id: str, frame_idx: int) -> AnnotationStore:
    annotations = list(store.videos.get(video_id, []))
    annotations = [a for a in annotations if a.frame_idx != frame_idx]
    updated_videos = {**store.videos, video_id: annotations}
    return AnnotationStore(videos=updated_videos)
