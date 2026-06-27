"""Model download module: fetches ONNX weights from HuggingFace Hub with caching."""
from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from .registry import ModelInfo, get_model_info

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deepfake_detector" / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_placeholder(checksum: str) -> bool:
    return checksum.startswith("PLACEHOLDER_")


class ChecksumMismatchError(ValueError):
    """Raised when a downloaded file's SHA256 does not match the manifest."""


def download(
    model_name: str,
    version: str = "latest",
    cache_dir: Path | str | None = None,
) -> Path:
    """Download *model_name* at *version* to *cache_dir* and return its local path.

    Idempotent: if the file already exists and the checksum matches (or the
    manifest still carries a placeholder checksum), the download is skipped.

    Args:
        model_name: Registry key, e.g. ``'efficientnet_b4_deepfake'``.
        version: Version tag, e.g. ``'v1.0'``, or ``'latest'`` to use the
            newest registered version (default).
        cache_dir: Override cache directory. Defaults to
            ``~/.cache/deepfake_detector/models/``.

    Returns:
        Absolute :class:`Path` to the cached model file.

    Raises:
        ModelNotFoundError: model not in registry.
        ChecksumMismatchError: downloaded file fails SHA256 verification.
    """
    info: ModelInfo = get_model_info(model_name, version)
    cache_root = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    dest = cache_root / info.filename

    # Cache-hit: skip download if file exists and checksum is valid.
    if dest.exists():
        if _is_placeholder(info.sha256):
            logger.debug("Cached model found (placeholder checksum, skipping verify): %s", dest)
            return dest
        if _sha256(dest) == info.sha256:
            logger.debug("Cached model checksum OK, skipping download: %s", dest)
            return dest
        logger.warning("Cached model checksum mismatch, re-downloading: %s", dest)

    logger.info(
        "Downloading %s %s (~%s MB) from %s …",
        model_name,
        info.version,
        info.size_mb,
        info.hf_repo,
    )

    tmp_path = hf_hub_download(
        repo_id=info.hf_repo,
        filename=info.filename,
        cache_dir=str(cache_root / ".hf_cache"),
    )
    shutil.copy2(tmp_path, dest)
    logger.info("Saved model to %s", dest)

    if not _is_placeholder(info.sha256):
        actual = _sha256(dest)
        if actual != info.sha256:
            dest.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"SHA256 mismatch for {model_name} {info.version}. "
                f"Expected {info.sha256}, got {actual}."
            )
        logger.debug("Checksum verified: %s", actual)

    return dest


def get_model_path(
    model_name: str,
    version: str = "latest",
    cache_dir: Path | str | None = None,
) -> Path:
    """Return the local path for *model_name*, downloading it if necessary.

    This is the primary entry-point for lazy model loading.

    Args:
        model_name: Registry key, e.g. ``'efficientnet_b4_deepfake'``.
        version: Version tag or ``'latest'`` (default).
        cache_dir: Override cache directory.

    Returns:
        Absolute :class:`Path` to the cached model file.
    """
    return download(model_name, version=version, cache_dir=cache_dir)
