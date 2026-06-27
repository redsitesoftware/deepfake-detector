"""deepfake_detector — real-time deepfake detection library.

Top-level imports are lazy so that ``import deepfake_detector`` is fast and
scripts that only need the models sub-package (e.g. download_models.py) do not
pull in heavy ML dependencies such as cv2 / mediapipe.
"""
from __future__ import annotations

from .types import DetectionResult, LivenessResult, TemporalResult

__all__ = [
    "DetectionResult",
    "Detector",
    "LivenessResult",
    "TemporalResult",
    "detect_file",
    "detect_frame",
    "detect_stream",
]


def __getattr__(name: str) -> object:
    if name in {"Detector", "detect_file", "detect_frame", "detect_stream"}:
        from .core import Detector, detect_file, detect_frame, detect_stream  # noqa: PLC0415

        globals().update(
            Detector=Detector,
            detect_file=detect_file,
            detect_frame=detect_frame,
            detect_stream=detect_stream,
        )
        return globals()[name]
    raise AttributeError(f"module 'deepfake_detector' has no attribute {name!r}")

