"""conftest.py — mock heavy ML/native dependencies so tests run without them installed."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _mock_module(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# Mock cv2 and all OpenCV-dependent imports before any project code loads them.
_cv2 = _mock_module(
    "cv2",
    VideoCapture=MagicMock,
    CAP_PROP_FPS=5,
    imencode=MagicMock(return_value=(True, MagicMock())),
)
sys.modules.setdefault("cv2", _cv2)

# Mock mediapipe
_mediapipe = _mock_module("mediapipe")
_mediapipe.solutions = _mock_module("mediapipe.solutions")
sys.modules.setdefault("mediapipe", _mediapipe)
sys.modules.setdefault("mediapipe.solutions", _mediapipe.solutions)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _mock_module("mediapipe.solutions.face_mesh"))
sys.modules.setdefault("mediapipe.solutions.face_detection", _mock_module("mediapipe.solutions.face_detection"))

# Mock torch (only needed for export script, not model management)
sys.modules.setdefault("torch", _mock_module("torch"))

# Mock scipy
sys.modules.setdefault("scipy", _mock_module("scipy"))
sys.modules.setdefault("scipy.spatial", _mock_module("scipy.spatial"))
sys.modules.setdefault("scipy.spatial.distance", _mock_module("scipy.spatial.distance"))
