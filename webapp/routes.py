from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from model.schemas import Annotation, BallAnnotation, BallSource, GameState
from webapp.ball_storage import (
    BALL_ANNOTATIONS_PATH,
    add_ball_annotation,
    load_ball_annotations,
    remove_ball_annotation,
    save_ball_annotations,
)
from webapp.detector import detect_ball
from webapp.storage import add_annotation, load_annotations, remove_annotation, save_annotations
from webapp.tracknet_detector import detect_all_frames_batch, detect_ball_tracknet
from webapp.video import extract_frame, get_video_info, list_videos

DATASET_DIR = Path("dataset/videos")
ANNOTATIONS_PATH = Path("dataset/annotations.json")

router = APIRouter(prefix="/api")


class AnnotationRequest(BaseModel):
    frame_idx: int
    label: GameState


class BallAnnotationRequest(BaseModel):
    frame_idx: int
    cx: float
    cy: float
    visibility: int = 1
    source: BallSource


# ── Videos ───────────────────────────────────────────────────────────────────

@router.get("/videos")
def get_videos() -> list[dict[str, str]]:
    return [e.model_dump() for e in list_videos(DATASET_DIR)]


@router.get("/videos/{video_id:path}/info")
def get_video_info_endpoint(video_id: str) -> dict[str, object]:
    try:
        info = get_video_info(DATASET_DIR, video_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return info.model_dump()


@router.get("/videos/{video_id:path}/frame/{frame_idx}")
def get_frame(video_id: str, frame_idx: int) -> Response:
    try:
        jpeg_bytes = extract_frame(DATASET_DIR, video_id, frame_idx)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=jpeg_bytes, media_type="image/jpeg")


# ── Event annotations (N/B/H) ────────────────────────────────────────────────

@router.get("/videos/{video_id:path}/annotations")
def get_annotations(video_id: str) -> list[dict[str, object]]:
    store = load_annotations(ANNOTATIONS_PATH)
    return [a.model_dump() for a in store.videos.get(video_id, [])]


@router.post("/videos/{video_id:path}/annotations")
def post_annotation(video_id: str, body: AnnotationRequest) -> dict[str, str]:
    store = load_annotations(ANNOTATIONS_PATH)
    store = add_annotation(store, video_id, Annotation(frame_idx=body.frame_idx, label=body.label))
    save_annotations(store, ANNOTATIONS_PATH)
    return {"status": "ok"}


@router.delete("/videos/{video_id:path}/annotations/{frame_idx}")
def delete_annotation(video_id: str, frame_idx: int) -> dict[str, str]:
    store = load_annotations(ANNOTATIONS_PATH)
    store = remove_annotation(store, video_id, frame_idx)
    save_annotations(store, ANNOTATIONS_PATH)
    return {"status": "ok"}


# ── Ball annotations ──────────────────────────────────────────────────────────

@router.get("/videos/{video_id:path}/ball")
def get_ball_annotations(video_id: str) -> list[dict[str, object]]:
    store = load_ball_annotations(BALL_ANNOTATIONS_PATH)
    return [a.model_dump() for a in store.videos.get(video_id, [])]


@router.post("/videos/{video_id:path}/ball")
def post_ball_annotation(video_id: str, body: BallAnnotationRequest) -> dict[str, str]:
    store = load_ball_annotations(BALL_ANNOTATIONS_PATH)
    ann = BallAnnotation(
        frame_idx=body.frame_idx,
        cx=body.cx, cy=body.cy,
        visibility=body.visibility,
        source=body.source,
    )
    store = add_ball_annotation(store, video_id, ann)
    save_ball_annotations(store, BALL_ANNOTATIONS_PATH)
    return {"status": "ok"}


@router.delete("/videos/{video_id:path}/ball/{frame_idx}")
def delete_ball_annotation(video_id: str, frame_idx: int) -> dict[str, str]:
    store = load_ball_annotations(BALL_ANNOTATIONS_PATH)
    store = remove_ball_annotation(store, video_id, frame_idx)
    save_ball_annotations(store, BALL_ANNOTATIONS_PATH)
    return {"status": "ok"}


@router.get("/videos/{video_id:path}/frame/{frame_idx}/detect_ball")
def detect_ball_endpoint(video_id: str, frame_idx: int) -> dict[str, object]:
    video_path = DATASET_DIR / video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

    try:
        info = get_video_info(DATASET_DIR, video_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Try TrackNet first
    tracknet_result = detect_ball_tracknet(video_path, frame_idx, info.width, info.height)
    if tracknet_result is not None:
        return {"found": True, "cx": tracknet_result.cx, "cy": tracknet_result.cy, "conf": tracknet_result.conf, "source": BallSource.tracknet}

    # Fallback to YOLO
    try:
        jpeg_bytes = extract_frame(DATASET_DIR, video_id, frame_idx)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = detect_ball(jpeg_bytes)
    if result is None:
        return {"found": False}
    return {"found": True, "cx": result.cx, "cy": result.cy, "conf": result.conf, "source": BallSource.yolo}


@router.post("/videos/{video_id:path}/detect_all")
def detect_all_endpoint(video_id: str) -> dict[str, object]:
    video_path = DATASET_DIR / video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

    try:
        info = get_video_info(DATASET_DIR, video_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    detections = detect_all_frames_batch(video_path, info.width, info.height)

    store = load_ball_annotations(BALL_ANNOTATIONS_PATH)
    for frame_idx, result in detections:
        ann = BallAnnotation(
            frame_idx=frame_idx,
            cx=result.cx,
            cy=result.cy,
            visibility=1,
            source=BallSource.tracknet,
        )
        store = add_ball_annotation(store, video_id, ann)
    save_ball_annotations(store, BALL_ANNOTATIONS_PATH)

    return {"detected": len(detections), "total": info.frame_count}
