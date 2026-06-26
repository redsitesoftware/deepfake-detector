from deepfake_detector.types import DetectionResult, LivenessResult, TemporalResult


def test_detection_result_defaults() -> None:
    liveness = LivenessResult(is_live=True, blink_rate=12.0, head_pose_variance=1.2)
    temporal = TemporalResult(
        temporal_score=0.4,
        flicker_score=0.3,
        drift_score=0.5,
        frames_analysed=16,
    )
    result = DetectionResult(
        is_fake=False,
        confidence=0.32,
        signals={"cnn": 0.2, "temporal": 0.4, "liveness": 0.0},
        face_detected=True,
        face_bbox=(10, 20, 30, 40),
        liveness=liveness,
        temporal=temporal,
        latency_ms=12.3,
    )

    assert result.model_version == "v0.1-bootstrap"
    assert result.liveness is liveness
    assert result.temporal is temporal
    assert result.face_bbox == (10, 20, 30, 40)
    assert result.signals["cnn"] == 0.2
