"""Model registry: maps model names + versions to HuggingFace Hub locations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_MANIFEST_PATH = Path(__file__).parent.parent.parent / "models" / "manifest.json"


@dataclass
class ModelInfo:
    """Metadata for a single versioned model."""

    name: str
    version: str
    hf_repo: str
    filename: str
    sha256: str
    format: str = "onnx_fp32"
    size_mb: Optional[float] = None
    trained_on: List[str] = field(default_factory=list)
    auc_celeb_df: Optional[float] = None
    description: Optional[str] = None

    @property
    def hf_url(self) -> str:
        """Canonical HuggingFace Hub URL for this model file."""
        return f"https://huggingface.co/{self.hf_repo}/resolve/main/{self.filename}"


def _load_manifest() -> Dict[str, Dict[str, dict]]:
    with open(_MANIFEST_PATH, "r") as f:
        return json.load(f)


def _build_registry() -> Dict[str, Dict[str, ModelInfo]]:
    raw = _load_manifest()
    registry: Dict[str, Dict[str, ModelInfo]] = {}
    for model_name, versions in raw.items():
        registry[model_name] = {}
        for version, meta in versions.items():
            registry[model_name][version] = ModelInfo(
                name=model_name,
                version=version,
                hf_repo=meta["hf_repo"],
                filename=meta["filename"],
                sha256=meta["sha256"],
                format=meta.get("format", "onnx_fp32"),
                size_mb=meta.get("size_mb"),
                trained_on=meta.get("trained_on", []),
                auc_celeb_df=meta.get("auc_celeb_df"),
                description=meta.get("description"),
            )
    return registry


MODEL_REGISTRY: Dict[str, Dict[str, ModelInfo]] = _build_registry()

DEFAULT_VERSIONS: Dict[str, str] = {
    "efficientnet_b4_deepfake": "v1.0",
    "face_detector_mtcnn": "v1.0",
}


def get_model_info(name: str, version: Optional[str] = None) -> ModelInfo:
    """Return ModelInfo for the given model name and version.

    Args:
        name: Model name (e.g. ``'efficientnet_b4_deepfake'``).
        version: Version string (e.g. ``'v1.0'``).  When omitted the default
            version for that model is used.

    Returns:
        :class:`ModelInfo` instance with HF repo, filename, and checksum.

    Raises:
        KeyError: If the model name or version is not found in the registry.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")

    resolved_version = version or DEFAULT_VERSIONS.get(name)
    if resolved_version is None or resolved_version not in MODEL_REGISTRY[name]:
        available_versions = ", ".join(sorted(MODEL_REGISTRY[name]))
        raise KeyError(
            f"Unknown version '{resolved_version}' for model '{name}'. "
            f"Available versions: {available_versions}"
        )

    return MODEL_REGISTRY[name][resolved_version]
