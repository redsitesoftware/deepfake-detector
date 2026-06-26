from __future__ import annotations

from pydantic import BaseModel, Field

from deepfake_detector.types import DetectionResult


class LivenessResponse(BaseModel):
    is_live: bool
    blink_rate: float
    head_pose_variance: float
    flags: list[str] = Field(default_factory=list)


class TemporalResponse(BaseModel):
    temporal_score: float
    flicker_score: float
    drift_score: float
    frames_analysed: int


class DetectionResponse(BaseModel):
    is_fake: bool
    confidence: float
    signals: dict[str, float | None]
    face_detected: bool
    face_bbox: tuple[int, int, int, int] | None = None
    liveness: LivenessResponse | None = None
    temporal: TemporalResponse | None = None
    latency_ms: float
    model_version: str


class VideoDetectionResponse(BaseModel):
    frame_count: int
    fake_frames: int
    average_confidence: float
    results: list[DetectionResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    warmed_up: bool
    warnings: list[str] = Field(default_factory=list)


class StreamFramePayload(BaseModel):
    frame: str
    frame_id: int


def detection_to_response(result: DetectionResult) -> DetectionResponse:
    return DetectionResponse(
        is_fake=result.is_fake,
        confidence=result.confidence,
        signals=result.signals,
        face_detected=result.face_detected,
        face_bbox=result.face_bbox,
        liveness=(
            LivenessResponse(
                is_live=result.liveness.is_live,
                blink_rate=result.liveness.blink_rate,
                head_pose_variance=result.liveness.head_pose_variance,
                flags=result.liveness.flags,
            )
            if result.liveness is not None
            else None
        ),
        temporal=(
            TemporalResponse(
                temporal_score=result.temporal.temporal_score,
                flicker_score=result.temporal.flicker_score,
                drift_score=result.temporal.drift_score,
                frames_analysed=result.temporal.frames_analysed,
            )
            if result.temporal is not None
            else None
        ),
        latency_ms=result.latency_ms,
        model_version=result.model_version,
    )
