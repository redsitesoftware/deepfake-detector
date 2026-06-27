"""Tests for model registry, download module, and checksum verification."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import deepfake_detector.models._download as _dl_mod  # direct submodule ref avoids name collision
from deepfake_detector.models import (
    MODEL_REGISTRY,
    ChecksumMismatchError,
    ModelNotFoundError,
    download,
    get_model_info,
    get_model_path,
)
from deepfake_detector.models.registry import ModelInfo

# ---------------------------------------------------------------------------
# Registry tests (#15)
# ---------------------------------------------------------------------------


def test_registry_contains_all_required_models() -> None:
    assert "efficientnet_b4_deepfake" in MODEL_REGISTRY
    assert "efficientnet_b4_deepfake_fp16" in MODEL_REGISTRY
    assert "face_detector_mtcnn" in MODEL_REGISTRY


def test_registry_get_model_info_returns_correct_fields() -> None:
    info = get_model_info("efficientnet_b4_deepfake", "v1.0")
    assert isinstance(info, ModelInfo)
    assert info.name == "efficientnet_b4_deepfake"
    assert info.version == "v1.0"
    assert info.hf_repo == "redsitesoftware/deepfake-detector-models"
    assert info.filename == "efficientnet_b4_v1.0.onnx"
    assert info.sha256  # non-empty
    assert "FaceForensics++" in info.trained_on


def test_registry_unknown_model_raises() -> None:
    with pytest.raises(ModelNotFoundError, match="Unknown model"):
        get_model_info("nonexistent_model_xyz")


def test_registry_unknown_version_raises() -> None:
    with pytest.raises(ModelNotFoundError, match="no version"):
        get_model_info("efficientnet_b4_deepfake", "v999.0")


def test_registry_face_detector_mtcnn_entry() -> None:
    info = get_model_info("face_detector_mtcnn", "v1.0")
    assert info.size_mb == 2
    assert info.format == "onnx_fp32"


# ---------------------------------------------------------------------------
# Download tests (#16) — all HF Hub calls are mocked
# ---------------------------------------------------------------------------


def _write_fake_file(path: Path, content: bytes = b"fake-onnx-weights") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@patch.object(_dl_mod, "hf_hub_download")
def test_download_fetches_when_not_cached(mock_hf: MagicMock, tmp_path: Path) -> None:
    fake_content = b"fake-onnx-model-data"
    tmp_file = tmp_path / "source.onnx"
    _write_fake_file(tmp_file, fake_content)
    mock_hf.return_value = str(tmp_file)

    # Patch registry sha256 to placeholder so checksum step is skipped
    info = get_model_info("face_detector_mtcnn", "v1.0")
    original_sha = info.sha256
    info.sha256 = "PLACEHOLDER_test"

    try:
        result = download("face_detector_mtcnn", version="v1.0", cache_dir=tmp_path)
        assert result.exists()
        mock_hf.assert_called_once()
    finally:
        info.sha256 = original_sha


@patch.object(_dl_mod, "hf_hub_download")
def test_download_skips_if_checksum_matches(mock_hf: MagicMock, tmp_path: Path) -> None:
    fake_content = b"cached-onnx-data"
    info = get_model_info("face_detector_mtcnn", "v1.0")
    original_sha = info.sha256

    # Pre-populate cache with a file whose checksum matches what we'll set
    dest = tmp_path / info.filename
    _write_fake_file(dest, fake_content)
    info.sha256 = _sha256(fake_content)  # make checksum match

    try:
        result = download("face_detector_mtcnn", version="v1.0", cache_dir=tmp_path)
        assert result == dest
        mock_hf.assert_not_called()  # no network call
    finally:
        info.sha256 = original_sha


@patch.object(_dl_mod, "hf_hub_download")
def test_download_raises_on_checksum_mismatch(mock_hf: MagicMock, tmp_path: Path) -> None:
    corrupt_content = b"this-is-corrupted"
    tmp_file = tmp_path / "source.onnx"
    _write_fake_file(tmp_file, corrupt_content)
    mock_hf.return_value = str(tmp_file)

    info = get_model_info("face_detector_mtcnn", "v1.0")
    original_sha = info.sha256
    info.sha256 = "a" * 64  # valid hex but wrong checksum

    try:
        with pytest.raises(ChecksumMismatchError, match="SHA256 mismatch"):
            download("face_detector_mtcnn", version="v1.0", cache_dir=tmp_path)
    finally:
        info.sha256 = original_sha


@patch.object(_dl_mod, "hf_hub_download")
def test_get_model_path_triggers_download(mock_hf: MagicMock, tmp_path: Path) -> None:
    fake_content = b"lazy-model-data"
    tmp_file = tmp_path / "source.onnx"
    _write_fake_file(tmp_file, fake_content)
    mock_hf.return_value = str(tmp_file)

    info = get_model_info("face_detector_mtcnn", "v1.0")
    original_sha = info.sha256
    info.sha256 = "PLACEHOLDER_lazy"

    try:
        path = get_model_path("face_detector_mtcnn", version="v1.0", cache_dir=tmp_path)
        assert path.exists()
        mock_hf.assert_called_once()
    finally:
        info.sha256 = original_sha


def test_download_unknown_model_raises() -> None:
    with pytest.raises(ModelNotFoundError):
        download("this_model_does_not_exist")


# ---------------------------------------------------------------------------
# core.py integration — detect_frame passes model_version (#16, #18)
# ---------------------------------------------------------------------------


@patch.object(_dl_mod, "hf_hub_download")
def test_detect_frame_uses_model_version_param(mock_hf: MagicMock, tmp_path: Path) -> None:
    """detect_frame(frame, model_version=...) must call get_model_path with that version."""
    fake_content = b"dummy-weights"
    tmp_file = tmp_path / "source.onnx"
    _write_fake_file(tmp_file, fake_content)
    mock_hf.return_value = str(tmp_file)

    info = get_model_info("efficientnet_b4_deepfake", "v1.0")
    original_sha = info.sha256
    info.sha256 = "PLACEHOLDER_core"

    try:
        with patch.object(_dl_mod, "_DEFAULT_CACHE_DIR", tmp_path):
            from deepfake_detector import core

            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)

            with patch.object(core, "Detector") as MockDetector:
                mock_instance = MagicMock()
                mock_instance.analyse.return_value = MagicMock()
                MockDetector.return_value = mock_instance

                core.detect_frame(dummy_frame, model_version="v1.0")

                # Ensure get_model_path was invoked (download was triggered)
                mock_hf.assert_called()
    finally:
        info.sha256 = original_sha
