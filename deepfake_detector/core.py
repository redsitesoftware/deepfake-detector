from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from deepfake_detector.cnn import CNNDetector
from deepfake_detector.liveness import LivenessAnalyser
from deepfake_detector.temporal import TemporalBuffer
from deepfake_detector.types import DetectionResult
from deepfake_detector.utils.face import detect_face

MODEL_VERSION = "v0.1-bootstrap"
FrameCallback = Callable[[DetectionResult], None]


def _weighted_mean(items: list[tuple[float | None, float]]) -> float:
    available = [(value, weight) for value, weight in items if value is not None]
    if not available:
        return 0.0
    numerator = sum(value * weight for value, weight in available)
    denominator = sum(weight for _, weight in available)
    if denominator == 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _coerce_video_source(source: str | int) -> str | int:
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def _frame_step(source_fps: float, target_fps: float | None) -> int:
    if target_fps is None or target_fps <= 0 or source_fps <= 0 or target_fps >= source_fps:
        return 1
    return max(int(round(source_fps / target_fps)), 1)


class Detector:
    def __init__(self, fps: float = 30.0, temporal_window_size: int = 16) -> None:
        self.cnn = CNNDetector()
        self.temporal_buffer = TemporalBuffer(window_size=temporal_window_size)
        self.liveness_analyser = LivenessAnalyser(fps=fps)
        self.model_version = MODEL_VERSION

    def warmup(self) -> None:
        self.cnn.warmup()
        self.liveness_analyser.warmup()

    def close(self) -> None:
        self.liveness_analyser.close()

    def analyse(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        if frame is None or frame.size == 0:
            return self._empty_result(start)

        face_bbox = detect_face(frame)
        if face_bbox is None:
            self.temporal_buffer.clear()
            self.liveness_analyser.reset()
            return self._empty_result(start)

        cnn_score = self.cnn.score(frame, face_bbox)
        temporal_result = self.temporal_buffer.push(frame, face_bbox)
        liveness_result = self.liveness_analyser.analyse(frame)
        liveness_score = None if liveness_result is None else float(not liveness_result.is_live)

        confidence = _weighted_mean(
            [
                (cnn_score, 0.5),
                (
                    temporal_result.temporal_score if temporal_result is not None else None,
                    0.3,
                ),
                (liveness_score, 0.2),
            ]
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(
            is_fake=confidence > 0.5,
            confidence=confidence,
            signals={
                "cnn": cnn_score,
                "temporal": temporal_result.temporal_score if temporal_result else None,
                "liveness": liveness_score,
            },
            face_detected=True,
            face_bbox=face_bbox,
            liveness=liveness_result,
            temporal=temporal_result,
            latency_ms=latency_ms,
            model_version=self.model_version,
        )

    def _empty_result(self, start: float) -> DetectionResult:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(
            is_fake=False,
            confidence=0.0,
            signals={"cnn": None, "temporal": None, "liveness": None},
            face_detected=False,
            face_bbox=None,
            liveness=None,
            temporal=None,
            latency_ms=latency_ms,
            model_version=self.model_version,
        )


def detect_frame(frame: np.ndarray) -> DetectionResult:
    detector = Detector()
    try:
        return detector.analyse(frame)
    finally:
        detector.close()


def detect_stream(
    source: str | int,
    callback: FrameCallback,
    fps_limit: float = 10.0,
) -> None:
    video_source = _coerce_video_source(source)
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps = fps_limit if fps_limit > 0 else source_fps
    detector = Detector(fps=target_fps)
    frame_step = _frame_step(source_fps, target_fps)
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            if (frame_index - 1) % frame_step != 0:
                continue
            callback(detector.analyse(frame))
    finally:
        detector.close()
        capture.release()


def detect_file(path: str | Path, fps_limit: float | None = None) -> list[DetectionResult]:
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps = fps_limit if fps_limit and fps_limit > 0 else source_fps
    detector = Detector(fps=target_fps)
    frame_step = _frame_step(source_fps, target_fps)
    frame_index = 0
    results: list[DetectionResult] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            if (frame_index - 1) % frame_step != 0:
                continue
            results.append(detector.analyse(frame))
    finally:
        detector.close()
        capture.release()
    return results
