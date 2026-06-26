from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from deepfake_detector.types import TemporalResult
from deepfake_detector.utils.face import crop_face


class TemporalBuffer:
    def __init__(self, window_size: int = 16) -> None:
        self.frames: deque[np.ndarray] = deque(maxlen=window_size)
        self.face_crops: deque[np.ndarray] = deque(maxlen=window_size)

    def clear(self) -> None:
        self.frames.clear()
        self.face_crops.clear()

    def push(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None,
    ) -> TemporalResult | None:
        if frame is None or frame.size == 0 or face_bbox is None:
            self.clear()
            return None

        face_crop = crop_face(frame, face_bbox, margin=0.2)
        if face_crop.size == 0:
            return None

        self.frames.append(frame.copy())
        self.face_crops.append(face_crop)
        if len(self.face_crops) < self.face_crops.maxlen:
            return None

        flicker_score = self._flicker_score()
        drift_score = self._drift_score()
        temporal_score = float(np.clip((flicker_score + drift_score) / 2.0, 0.0, 1.0))
        return TemporalResult(
            temporal_score=temporal_score,
            flicker_score=flicker_score,
            drift_score=drift_score,
            frames_analysed=len(self.face_crops),
        )

    def _normalised_face_crops(self) -> list[np.ndarray]:
        return [
            cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
            for crop in self.face_crops
            if crop.size > 0
        ]

    def _flicker_score(self) -> float:
        crops = self._normalised_face_crops()
        if len(crops) < 2:
            return 0.0

        edge_maps = [
            cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150).astype(np.float32)
            for crop in crops
        ]
        diffs = [
            float(np.mean(cv2.absdiff(edge_maps[index], edge_maps[index - 1])) / 255.0)
            for index in range(1, len(edge_maps))
        ]
        return float(np.clip(np.mean(diffs), 0.0, 1.0))

    def _drift_score(self) -> float:
        crops = self._normalised_face_crops()
        if len(crops) < 2:
            return 0.0

        histograms = []
        for crop in crops:
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist(
                [hsv_crop],
                [0, 1, 2],
                None,
                [8, 8, 8],
                [0, 180, 0, 256, 0, 256],
            )
            cv2.normalize(histogram, histogram)
            histograms.append(histogram)

        drift_scores = [
            float(
                cv2.compareHist(
                    histograms[index - 1],
                    histograms[index],
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
            for index in range(1, len(histograms))
        ]
        return float(np.clip(np.mean(drift_scores), 0.0, 1.0))
