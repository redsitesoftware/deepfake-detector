"""Model registry: loads the manifest and provides lookup helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MANIFEST_PATH = Path(__file__).parent.parent.parent / "models" / "manifest.json"


@dataclass
class ModelInfo:
    name: str
    version: str
    hf_repo: str
    filename: str
    sha256: str
    format: str
    size_mb: int
    trained_on: list[str]
    auc_celeb_df: float | None
    description: str


class ModelNotFoundError(KeyError):
    """Raised when a requested model name or version is not in the registry."""


def _load_manifest() -> dict[str, Any]:
    with _MANIFEST_PATH.open() as fh:
        return json.load(fh)


# Eagerly load once at module import — the JSON is tiny.
MODEL_REGISTRY: dict[str, dict[str, ModelInfo]] = {}

for _model_name, _versions in _load_manifest().items():
    MODEL_REGISTRY[_model_name] = {}
    for _version, _meta in _versions.items():
        MODEL_REGISTRY[_model_name][_version] = ModelInfo(
            name=_model_name,
            version=_version,
            hf_repo=_meta["hf_repo"],
            filename=_meta["filename"],
            sha256=_meta["sha256"],
            format=_meta["format"],
            size_mb=_meta["size_mb"],
            trained_on=_meta["trained_on"],
            auc_celeb_df=_meta.get("auc_celeb_df"),
            description=_meta.get("description", ""),
        )


def _resolve_version(model_name: str, version: str) -> str:
    """Resolve ``'latest'`` to the highest version tag in the registry."""
    if version != "latest":
        return version
    versions = MODEL_REGISTRY.get(model_name, {})
    if not versions:
        return version
    return sorted(versions)[-1]


def get_model_info(model_name: str, version: str = "latest") -> ModelInfo:
    """Return ModelInfo for *model_name* at *version*.

    Pass ``version='latest'`` (the default) to resolve the newest available
    version automatically.

    Raises:
        ModelNotFoundError: if model_name or version is absent from the registry.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ModelNotFoundError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    resolved = _resolve_version(model_name, version)
    versions = MODEL_REGISTRY[model_name]
    if resolved not in versions:
        available = ", ".join(versions)
        raise ModelNotFoundError(
            f"Model '{model_name}' has no version '{resolved}'. Available: {available}"
        )
    return versions[resolved]
