from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from api.schemas import (
    DetectionResponse,
    HealthResponse,
    VideoDetectionResponse,
    detection_to_response,
)
from deepfake_detector import Detector, detect_file
from deepfake_detector.core import MODEL_VERSION

router = APIRouter()
UPLOAD_FILE = File(...)


def _decode_image_bytes(payload: bytes) -> np.ndarray:
    buffer = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image payload")
    return frame


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = getattr(request.app.state, "detector_status", {"warmed_up": False, "warnings": []})
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        warmed_up=bool(state.get("warmed_up", False)),
        warnings=list(state.get("warnings", [])),
    )


@router.post("/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = UPLOAD_FILE) -> DetectionResponse:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        frame = _decode_image_bytes(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detector = Detector()
    try:
        return detection_to_response(detector.analyse(frame))
    finally:
        detector.close()


@router.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_video(file: UploadFile = UPLOAD_FILE) -> VideoDetectionResponse:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty video upload")

    runtime_dir = Path.cwd() / ".runtime_uploads"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.mp4").name
    temp_path = runtime_dir / f"{uuid4().hex}-{safe_name}"
    try:
        temp_path.write_bytes(payload)
        results = detect_file(temp_path)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    fake_frames = sum(1 for result in results if result.is_fake)
    average_confidence = (
        float(np.mean([result.confidence for result in results])) if results else 0.0
    )
    return VideoDetectionResponse(
        frame_count=len(results),
        fake_frames=fake_frames,
        average_confidence=average_confidence,
        results=[detection_to_response(result) for result in results],
    )
