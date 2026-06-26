from __future__ import annotations

import cv2
import numpy as np

_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
        if _FACE_CASCADE.empty():
            raise RuntimeError(f"Unable to load Haar cascade from {cascade_path}")
    return _FACE_CASCADE


def detect_face(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    if frame is None or frame.size == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _get_face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda bbox: int(bbox[2]) * int(bbox[3]))
    return int(x), int(y), int(w), int(h)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    margin: float = 0.2,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x, y, w, h = bbox
    x_margin = int(w * margin)
    y_margin = int(h * margin)

    x1 = max(0, x - x_margin)
    y1 = max(0, y - y_margin)
    x2 = min(width, x + w + x_margin)
    y2 = min(height, y + h + y_margin)

    return x1, y1, x2 - x1, y2 - y1


def crop_face(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: float = 0.2,
) -> np.ndarray:
    x, y, w, h = expand_bbox(bbox, frame.shape, margin=margin)
    return frame[y : y + h, x : x + w].copy()
