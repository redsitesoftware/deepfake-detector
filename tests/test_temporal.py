import numpy as np

from deepfake_detector.temporal.analyser import TemporalBuffer


def test_temporal_buffer_identical_frames_have_low_flicker() -> None:
    buffer = TemporalBuffer(window_size=16)
    frame = np.full((96, 96, 3), 127, dtype=np.uint8)
    bbox = (16, 16, 48, 48)

    result = None
    for _ in range(16):
        result = buffer.push(frame, bbox)

    assert result is not None
    assert result.frames_analysed == 16
    assert result.flicker_score <= 0.01
    assert result.temporal_score <= 0.05


def test_temporal_buffer_random_frames_have_high_flicker() -> None:
    rng = np.random.default_rng(7)
    buffer = TemporalBuffer(window_size=16)
    bbox = (16, 16, 48, 48)

    result = None
    for _ in range(16):
        frame = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
        result = buffer.push(frame, bbox)

    assert result is not None
    assert result.flicker_score >= 0.15
