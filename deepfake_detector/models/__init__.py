"""deepfake_detector.models — model registry, download, and path resolution."""
from ._download import ChecksumMismatchError, download, get_model_path
from .registry import MODEL_REGISTRY, ModelInfo, ModelNotFoundError, get_model_info

__all__ = [
    "MODEL_REGISTRY",
    "ChecksumMismatchError",
    "ModelInfo",
    "ModelNotFoundError",
    "download",
    "get_model_info",
    "get_model_path",
]
