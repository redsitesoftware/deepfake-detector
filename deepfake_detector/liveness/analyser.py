from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deepfake_detector.types import LivenessResult

_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _ensure_model() -> None:
    if _MODEL_PATH.exists():
        return
    import urllib.request
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[liveness] Downloading face landmarker model to {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[liveness] Model downloaded.")

LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
HEAD_POSE_INDICES = (1, 152, 263, 33, 287, 57)
HEAD_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


def _eye_aspect_ratio(points: np.ndarray) -> float:
    vertical_a = np.linalg.norm(points[1] - points[5])
    vertical_b = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal == 0:
        return 0.0
    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def _rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> np.ndarray:
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x_angle = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y_angle = math.atan2(-rotation_matrix[2, 0], sy)
        z_angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x_angle = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y_angle = math.atan2(-rotation_matrix[2, 0], sy)
        z_angle = 0.0

    return np.degrees(np.array([x_angle, y_angle, z_angle], dtype=np.float64))


class LivenessAnalyser:
    def __init__(
        self,
        fps: float = 30.0,
        blink_window_frames: int = 90,
        pose_window_frames: int = 30,
        ear_threshold: float = 0.25,
    ) -> None:
        self.fps = fps
        self.blink_window_frames = blink_window_frames
        self.pose_window_frames = pose_window_frames
        self.ear_threshold = ear_threshold
        self.frame_index = 0
        self.pose_history: deque[np.ndarray] = deque(maxlen=pose_window_frames)
        self.blink_frames: deque[int] = deque()
        self._eye_closed = False
        self._face_mesh: Any | None = None

    def _get_face_mesh(self):
        if self._face_mesh is None:
            _ensure_model()
            import mediapipe as mp

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return self._face_mesh

    def close(self) -> None:
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
            self._face_mesh = None

    def reset(self) -> None:
        self.frame_index = 0
        self.pose_history.clear()
        self.blink_frames.clear()
        self._eye_closed = False

    def warmup(self) -> None:
        self._get_face_mesh()

    def analyse(self, frame: np.ndarray) -> LivenessResult | None:
        if frame is None or frame.size == 0:
            return None

        self.frame_index += 1
        import mediapipe as mp

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        face_mesh = self._get_face_mesh()
        detection = face_mesh.detect(mp_image)
        if not detection.face_landmarks:
            return None

        height, width = frame.shape[:2]
        raw_landmarks = detection.face_landmarks[0]
        landmarks_2d = np.array(
            [(lm.x * width, lm.y * height) for lm in raw_landmarks],
            dtype=np.float64,
        )

        ear = self._mean_eye_aspect_ratio(landmarks_2d)
        self._update_blinks(ear)
        pose_angles = self._solve_head_pose(landmarks_2d, width, height)
        if pose_angles is not None:
            self.pose_history.append(pose_angles)

        if len(self.pose_history) < self.pose_window_frames:
            return None

        blink_rate = self._blink_rate()
        head_pose_variance = self._head_pose_variance()
        flags: list[str] = []

        if self.frame_index >= self.blink_window_frames:
            # Only flag truly zero-blink windows — normal blink rate is
            # 15-20/min but people staring at a screen blink less.
            # Flag only if < 1 blink per minute (nearly absent).
            if blink_rate < 1.0:
                flags.append("blink_rate_too_low")
            elif blink_rate > 60.0:
                flags.append("blink_rate_too_high")

        # Require very low variance to flag — deepfakes tend to be completely
        # static or have robotic motion. A sitting human will have > 0.15°
        # of natural micro-movement even when trying to stay still.
        if head_pose_variance < 0.15:
            flags.append("static_head_pose")

        return LivenessResult(
            is_live=not flags,
            blink_rate=blink_rate,
            head_pose_variance=head_pose_variance,
            flags=flags,
        )

    def _mean_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        left_eye = landmarks[list(LEFT_EYE_INDICES)]
        right_eye = landmarks[list(RIGHT_EYE_INDICES)]
        return (_eye_aspect_ratio(left_eye) + _eye_aspect_ratio(right_eye)) / 2.0

    def _update_blinks(self, ear: float) -> None:
        is_closed = ear < self.ear_threshold
        if is_closed and not self._eye_closed:
            self._eye_closed = True
        elif not is_closed and self._eye_closed:
            self._eye_closed = False
            self.blink_frames.append(self.frame_index)

        cutoff = self.frame_index - self.blink_window_frames
        while self.blink_frames and self.blink_frames[0] <= cutoff:
            self.blink_frames.popleft()

    def _blink_rate(self) -> float:
        frames_considered = min(self.frame_index, self.blink_window_frames)
        if frames_considered <= 0 or self.fps <= 0:
            return 0.0
        minutes = frames_considered / self.fps / 60.0
        if minutes == 0:
            return 0.0
        return len(self.blink_frames) / minutes

    def _head_pose_variance(self) -> float:
        pose_array = np.array(self.pose_history, dtype=np.float64)
        stddev = np.std(pose_array, axis=0)
        return float(np.mean(stddev))

    def _solve_head_pose(
        self,
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray | None:
        image_points = landmarks[list(HEAD_POSE_INDICES)].astype(np.float64)
        focal_length = float(frame_width)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, frame_width / 2.0],
                [0.0, focal_length, frame_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vector, _translation_vector = cv2.solvePnP(
            HEAD_MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return _rotation_matrix_to_euler_angles(rotation_matrix)
