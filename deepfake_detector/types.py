from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LivenessResult:
    is_live: bool
    blink_rate: float
    head_pose_variance: float
    flags: list[str] = field(default_factory=list)


@dataclass
class TemporalResult:
    temporal_score: float
    flicker_score: float
    drift_score: float
    frames_analysed: int


@dataclass
class DetectionResult:
    is_fake: bool
    confidence: float
    signals: dict[str, float | None]
    face_detected: bool
    face_bbox: tuple[int, int, int, int] | None
    liveness: LivenessResult | None
    temporal: TemporalResult | None
    latency_ms: float
    model_version: str = "v0.1-bootstrap"
