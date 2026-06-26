import numpy as np

from deepfake_detector.liveness.analyser import _eye_aspect_ratio


def test_eye_aspect_ratio_for_open_eye() -> None:
    eye = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 2.0],
            [4.0, 0.0],
            [3.0, -2.0],
            [1.0, -2.0],
        ],
        dtype=np.float64,
    )

    assert _eye_aspect_ratio(eye) > 0.25


def test_eye_aspect_ratio_for_closed_eye() -> None:
    eye = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.1],
            [3.0, 0.1],
            [4.0, 0.0],
            [3.0, -0.1],
            [1.0, -0.1],
        ],
        dtype=np.float64,
    )

    assert _eye_aspect_ratio(eye) < 0.25
