from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from deepfake_detector.types import TemporalResult
from deepfake_detector.utils.face import crop_face

# Rolling window of recent raw flicker values used to build a baseline.
# We only flag a frame as temporally anomalous if it's significantly
# above the baseline — natural head motion and lighting changes create
# a non-zero "real face" flicker that we must not treat as fake.
_BASELINE_WINDOW = 60   # frames of history for baseline estimate
_ANOMALY_THRESHOLD = 1.8  # must be this many × baseline to score > 0.5


class TemporalBuffer:
    def __init__(self, window_size: int = 8) -> None:
        self.frames: deque[np.ndarray] = deque(maxlen=window_size)
        self.face_crops: deque[np.ndarray] = deque(maxlen=window_size)
        # Baseline: rolling history of raw flicker values for calibration
        self._raw_flicker_history: deque[float] = deque(maxlen=_BASELINE_WINDOW)
        self._raw_drift_history: deque[float]   = deque(maxlen=_BASELINE_WINDOW)

    def clear(self) -> None:
        self.frames.clear()
        self.face_crops.clear()
        # Do NOT clear history on toggle — we want the baseline to persist
        # so the detector can immediately compare against it after toggling.

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

        raw_flicker = self._raw_flicker_score()
        raw_drift   = self._raw_drift_score()

        # Accumulate into rolling history
        self._raw_flicker_history.append(raw_flicker)
        self._raw_drift_history.append(raw_drift)

        # Anomaly score: how much above the recent baseline is this frame?
        temporal_score = self._anomaly_score(
            raw_flicker, raw_drift,
            self._raw_flicker_history, self._raw_drift_history,
        )

        return TemporalResult(
            temporal_score=temporal_score,
            flicker_score=raw_flicker,
            drift_score=raw_drift,
            frames_analysed=len(self.face_crops),
        )

    def _anomaly_score(
        self,
        flicker: float, drift: float,
        flicker_history: deque, drift_history: deque,
    ) -> float:
        """Return 0–1 anomaly score relative to the rolling real-face baseline."""
        if len(flicker_history) < 8:
            # Not enough history yet — can't score
            return 0.0

        baseline_flicker = float(np.median(flicker_history))
        baseline_drift   = float(np.median(drift_history))

        def _relative(val: float, baseline: float) -> float:
            if baseline < 1e-6:
                return 0.0
            ratio = val / baseline
            # Sigmoid centred on _ANOMALY_THRESHOLD: score 0.5 at threshold,
            # approaches 1.0 at 3× baseline, approaches 0 at 1× baseline.
            return float(1.0 / (1.0 + np.exp(-3.0 * (ratio - _ANOMALY_THRESHOLD))))

        score = (_relative(flicker, baseline_flicker) * 0.6
                 + _relative(drift, baseline_drift)   * 0.4)
        return float(np.clip(score, 0.0, 1.0))

    def _normalised_face_crops(self) -> list[np.ndarray]:
        return [
            cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
            for crop in self.face_crops
            if crop.size > 0
        ]

    def _raw_flicker_score(self) -> float:
        crops = self._normalised_face_crops()
        if len(crops) < 2:
            return 0.0
        edge_maps = [
            cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150).astype(np.float32)
            for crop in crops
        ]
        diffs = [
            float(np.mean(cv2.absdiff(edge_maps[i], edge_maps[i - 1])) / 255.0)
            for i in range(1, len(edge_maps))
        ]
        return float(np.clip(np.mean(diffs), 0.0, 1.0))

    def _raw_drift_score(self) -> float:
        crops = self._normalised_face_crops()
        if len(crops) < 2:
            return 0.0
        histograms = []
        for crop in crops:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(h, h)
            histograms.append(h)
        drift_scores = [
            float(cv2.compareHist(histograms[i - 1], histograms[i], cv2.HISTCMP_BHATTACHARYYA))
            for i in range(1, len(histograms))
        ]
        return float(np.clip(np.mean(drift_scores), 0.0, 1.0))
