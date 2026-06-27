"""Pytest configuration: stub out heavy C-extension modules so tests that only
exercise pure-Python code can run without a full ML environment."""
from __future__ import annotations

import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    stub = types.ModuleType(name)
    stub.__spec__ = None  # type: ignore[assignment]
    return stub


# Stub cv2 and mediapipe only if they are not already installed.
for _mod in ("cv2", "mediapipe"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = _make_stub(_mod)
