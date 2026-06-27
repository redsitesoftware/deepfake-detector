"""deepfake_detector.models — model registry and metadata."""

from .registry import (
    DEFAULT_VERSIONS,
    MODEL_REGISTRY,
    ModelInfo,
    get_model_info,
)

__all__ = [
    "DEFAULT_VERSIONS",
    "MODEL_REGISTRY",
    "ModelInfo",
    "get_model_info",
]
