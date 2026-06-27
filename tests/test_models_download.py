"""Tests for deepfake_detector.models.download and registry."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepfake_detector.models import (
    ChecksumMismatchError,
    ModelNotFoundError,
    download,
    get_model_info,
    get_model_path,
)
from deepfake_detector.models.registry import MODEL_REGISTRY, _resolve_version

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_file(path: Path, content: bytes = b"fake-model-weights") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_model_registry_loaded(self):
        assert "efficientnet_b4_deepfake" in MODEL_REGISTRY

    def test_get_model_info_explicit_version(self):
        info = get_model_info("efficientnet_b4_deepfake", "v1.0")
        assert info.name == "efficientnet_b4_deepfake"
        assert info.version == "v1.0"
        assert info.hf_repo == "redsitesoftware/deepfake-detector-models"
        assert info.filename.endswith(".onnx")
        assert info.sha256.startswith("PLACEHOLDER_")

    def test_get_model_info_latest_resolves(self):
        info = get_model_info("efficientnet_b4_deepfake", "latest")
        assert info.version == "v1.0"

    def test_get_model_info_default_version_is_latest(self):
        info_default = get_model_info("efficientnet_b4_deepfake")
        info_latest = get_model_info("efficientnet_b4_deepfake", "latest")
        assert info_default.version == info_latest.version

    def test_get_model_info_unknown_model(self):
        with pytest.raises(ModelNotFoundError):
            get_model_info("nonexistent_model")

    def test_get_model_info_unknown_version(self):
        with pytest.raises(ModelNotFoundError):
            get_model_info("efficientnet_b4_deepfake", "v99.0")

    def test_resolve_version_latest(self):
        resolved = _resolve_version("efficientnet_b4_deepfake", "latest")
        assert resolved == "v1.0"

    def test_resolve_version_explicit_passthrough(self):
        resolved = _resolve_version("efficientnet_b4_deepfake", "v1.0")
        assert resolved == "v1.0"

    def test_all_registered_models_have_required_fields(self):
        for model_name, versions in MODEL_REGISTRY.items():
            for version, info in versions.items():
                assert info.hf_repo, f"{model_name}/{version} missing hf_repo"
                assert info.filename, f"{model_name}/{version} missing filename"
                assert info.sha256, f"{model_name}/{version} missing sha256"


# ---------------------------------------------------------------------------
# Download tests — HF Hub is mocked throughout
# ---------------------------------------------------------------------------

@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "model_cache"


@pytest.fixture()
def model_content() -> bytes:
    return b"mock-onnx-model-content"


def _mock_hf_download(tmp_path: Path, content: bytes):
    """Return a side_effect function that writes *content* to a temp file."""
    def _side_effect(repo_id: str, filename: str, cache_dir: str, **kwargs) -> str:
        dest = Path(cache_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return str(dest)
    return _side_effect


class TestDownloadPlaceholderChecksum:
    """When manifest has PLACEHOLDER_ checksum, download should not verify."""

    def test_download_fetches_file(self, tmp_path, cache_dir, model_content):
        with patch(
            "deepfake_detector.models._download.hf_hub_download",
            side_effect=_mock_hf_download(tmp_path, model_content),
        ):
            path = download("efficientnet_b4_deepfake", version="v1.0", cache_dir=cache_dir)

        assert path.exists()
        assert path.read_bytes() == model_content

    def test_download_returns_path_object(self, tmp_path, cache_dir, model_content):
        with patch(
            "deepfake_detector.models._download.hf_hub_download",
            side_effect=_mock_hf_download(tmp_path, model_content),
        ):
            path = download("efficientnet_b4_deepfake", version="latest", cache_dir=cache_dir)

        assert isinstance(path, Path)

    def test_download_places_file_in_cache_dir(self, tmp_path, cache_dir, model_content):
        with patch(
            "deepfake_detector.models._download.hf_hub_download",
            side_effect=_mock_hf_download(tmp_path, model_content),
        ):
            path = download("efficientnet_b4_deepfake", cache_dir=cache_dir)

        assert path.parent == cache_dir

    def test_idempotent_skips_hf_when_file_cached(self, tmp_path, cache_dir, model_content):
        """Second call must NOT hit HF Hub if file already exists (placeholder checksum)."""
        info = get_model_info("efficientnet_b4_deepfake", "v1.0")
        cached = cache_dir / info.filename
        _write_file(cached, model_content)

        mock_hf = MagicMock()
        with patch("deepfake_detector.models._download.hf_hub_download", mock_hf):
            path = download("efficientnet_b4_deepfake", version="v1.0", cache_dir=cache_dir)

        mock_hf.assert_not_called()
        assert path == cached

    def test_default_cache_dir_used_when_none_given(self, tmp_path, model_content, monkeypatch):
        default_cache = tmp_path / ".cache" / "deepfake_detector" / "models"

        import deepfake_detector.models._download as dl_module

        monkeypatch.setattr(dl_module, "_DEFAULT_CACHE_DIR", default_cache)

        with patch(
            "deepfake_detector.models._download.hf_hub_download",
            side_effect=_mock_hf_download(tmp_path, model_content),
        ):
            path = download("efficientnet_b4_deepfake", version="v1.0")

        assert path.parent == default_cache


class TestDownloadRealChecksum:
    """When manifest has a real SHA256, verify it after download."""

    def test_checksum_verified_on_download(self, tmp_path, cache_dir, model_content):
        real_sha = _sha256(model_content)

        info = get_model_info("efficientnet_b4_deepfake", "v1.0")
        # Temporarily patch the registry entry's sha256
        original_sha = info.sha256
        info.sha256 = real_sha
        try:
            with patch(
                "deepfake_detector.models._download.hf_hub_download",
                side_effect=_mock_hf_download(tmp_path, model_content),
            ):
                path = download("efficientnet_b4_deepfake", version="v1.0", cache_dir=cache_dir)
            assert path.exists()
        finally:
            info.sha256 = original_sha

    def test_checksum_mismatch_raises_and_removes_file(self, tmp_path, cache_dir, model_content):
        info = get_model_info("efficientnet_b4_deepfake", "v1.0")
        original_sha = info.sha256
        # Set a known-bad real (non-placeholder) checksum
        info.sha256 = "a" * 64
        try:
            with patch(
                "deepfake_detector.models._download.hf_hub_download",
                side_effect=_mock_hf_download(tmp_path, model_content),
            ):
                with pytest.raises(ChecksumMismatchError):
                    download("efficientnet_b4_deepfake", version="v1.0", cache_dir=cache_dir)

            # File must be removed on mismatch
            dest = cache_dir / info.filename
            assert not dest.exists()
        finally:
            info.sha256 = original_sha

    def test_cached_file_with_correct_checksum_skips_download(
        self, tmp_path, cache_dir, model_content
    ):
        real_sha = _sha256(model_content)
        info = get_model_info("efficientnet_b4_deepfake", "v1.0")
        original_sha = info.sha256
        info.sha256 = real_sha

        cached = cache_dir / info.filename
        _write_file(cached, model_content)

        mock_hf = MagicMock()
        try:
            with patch("deepfake_detector.models._download.hf_hub_download", mock_hf):
                path = download("efficientnet_b4_deepfake", version="v1.0", cache_dir=cache_dir)
            mock_hf.assert_not_called()
            assert path == cached
        finally:
            info.sha256 = original_sha

    def test_cached_file_with_wrong_checksum_redownloads(
        self, tmp_path, cache_dir, model_content
    ):
        real_sha = _sha256(model_content)
        info = get_model_info("efficientnet_b4_deepfake", "v1.0")
        original_sha = info.sha256
        info.sha256 = real_sha

        cached = cache_dir / info.filename
        _write_file(cached, b"stale-corrupted-content")

        try:
            with patch(
                "deepfake_detector.models._download.hf_hub_download",
                side_effect=_mock_hf_download(tmp_path, model_content),
            ) as mock_hf:
                path = download("efficientnet_b4_deepfake", version="v1.0", cache_dir=cache_dir)
            mock_hf.assert_called_once()
            assert path.read_bytes() == model_content
        finally:
            info.sha256 = original_sha


# ---------------------------------------------------------------------------
# get_model_path tests
# ---------------------------------------------------------------------------

class TestGetModelPath:
    def test_get_model_path_calls_download(self, tmp_path, cache_dir, model_content):
        with patch(
            "deepfake_detector.models._download.hf_hub_download",
            side_effect=_mock_hf_download(tmp_path, model_content),
        ):
            path = get_model_path("efficientnet_b4_deepfake", version="v1.0", cache_dir=cache_dir)

        assert path.exists()

    def test_get_model_path_unknown_model(self, cache_dir):
        with pytest.raises(ModelNotFoundError):
            get_model_path("no_such_model", cache_dir=cache_dir)
