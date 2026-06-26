from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from deepfake_detector.utils.face import crop_face

LOGGER = logging.getLogger(__name__)
MODEL_ID = "dima806/deepfake_vs_real_image_detection"
_MODEL_NOTE = "Bootstrap model. Replace with trained EfficientNet-B4 (see issue #2)"
_PIPELINE = None


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline

        _PIPELINE = pipeline("image-classification", model=MODEL_ID)
    return _PIPELINE


class CNNDetector:
    model_id = MODEL_ID
    model_note = _MODEL_NOTE

    def warmup(self) -> None:
        _load_pipeline()

    def score(self, frame: np.ndarray, face_bbox: tuple[int, int, int, int]) -> float | None:
        try:
            face_crop = crop_face(frame, face_bbox, margin=0.2)
            if face_crop.size == 0:
                return None

            resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
            rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_face)
            predictions = _load_pipeline()(image, top_k=None)
        except Exception as exc:
            LOGGER.exception("CNN detector failed: %s", exc)
            return None

        score_by_label = {
            str(item["label"]).strip().lower(): float(item["score"])
            for item in predictions
        }
        if "fake" in score_by_label:
            return float(np.clip(score_by_label["fake"], 0.0, 1.0))
        if "real" in score_by_label:
            return float(np.clip(1.0 - score_by_label["real"], 0.0, 1.0))
        return None
