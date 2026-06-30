from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from deepfake_detector.utils.face import crop_face

LOGGER = logging.getLogger(__name__)

# ImageNet normalisation constants (same as training pipeline)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 380   # EfficientNet-B4 native resolution

_MODEL_NAME    = "efficientnet_b4_deepfake"
_MODEL_VERSION = "v1.0"
_MODEL_NOTE    = (
    "EfficientNet-B4 trained on FF++ / DF40. "
    "Run scripts/finetune_pretrained.py + scripts/export_onnx.py to update."
)

_SESSION: object | None = None   # onnxruntime.InferenceSession, lazily loaded
_INPUT_NAME: str | None = None
_N_OUTPUTS: int | None  = None   # 1 (BCE/sigmoid) or 2 (CE/softmax)


def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """Resize + ImageNet-normalise a BGR face crop → (1, 3, H, W) float32."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (_INPUT_SIZE, _INPUT_SIZE):
        rgb = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = rgb.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD                         # (H, W, 3)
    return img.transpose(2, 0, 1)[np.newaxis, ...]     # (1, 3, H, W)


def _resolve_model_path() -> Path | None:
    """Return path to ONNX model file, downloading from HuggingFace if needed.

    Search order:
      1. DEEPFAKE_DETECTOR_MODEL env var (absolute path override)
      2. models/ in repo root (local development)
      3. HuggingFace Hub cache via model registry
    """
    # 1. Explicit override
    env_path = os.environ.get("DEEPFAKE_DETECTOR_MODEL")
    if env_path and Path(env_path).exists():
        LOGGER.debug("Using model from DEEPFAKE_DETECTOR_MODEL: %s", env_path)
        return Path(env_path)

    # 2. Local models/ directory (dev mode — after export_onnx.py)
    repo_root = Path(__file__).parent.parent.parent
    for candidate in sorted((repo_root / "models").glob("efficientnet_b4_*.onnx")):
        if "_fp16" not in candidate.name:
            LOGGER.debug("Using local ONNX: %s", candidate)
            return candidate

    # 3. HuggingFace Hub download
    try:
        from deepfake_detector.models._download import get_model_path
        return get_model_path(_MODEL_NAME, _MODEL_VERSION)
    except Exception as exc:
        LOGGER.warning("Could not resolve model from registry: %s", exc)
        return None


def _load_session() -> bool:
    """Load the ONNX inference session. Returns True on success."""
    global _SESSION, _INPUT_NAME, _N_OUTPUTS
    if _SESSION is not None:
        return True

    try:
        import onnxruntime as ort
    except ImportError:
        LOGGER.warning("onnxruntime not installed — CNN detector disabled. "
                       "Install: pip install onnxruntime")
        return False

    model_path = _resolve_model_path()
    if model_path is None:
        LOGGER.warning(
            "CNN model not found. Run scripts/export_onnx.py to generate it, "
            "or set DEEPFAKE_DETECTOR_MODEL=/path/to/model.onnx"
        )
        return False

    try:
        providers = ["CPUExecutionProvider"]
        # Use CoreML on Apple Silicon if available
        if "CoreMLExecutionProvider" in ort.get_available_providers():
            providers = ["CoreMLExecutionProvider"] + providers

        sess = ort.InferenceSession(str(model_path), providers=providers)
        _INPUT_NAME = sess.get_inputs()[0].name
        _N_OUTPUTS  = sess.get_outputs()[0].shape[-1]  # 1 (sigmoid) or 2 (softmax)
        _SESSION    = sess
        LOGGER.info("CNN model loaded: %s  outputs=%d  providers=%s",
                    model_path.name, _N_OUTPUTS, providers)
        return True
    except Exception as exc:
        LOGGER.error("Failed to load CNN model from %s: %s", model_path, exc)
        return False


class CNNDetector:
    """Frame-level deepfake detector using EfficientNet-B4 ONNX model.

    Returns P(fake) ∈ [0, 1].  None if model unavailable or face crop fails.

    The model expects a 380×380 face crop, ImageNet-normalised.
    Architecture: EfficientNet-B4 (efficientnet_pytorch backbone).
    Output: 2-class softmax [P(real), P(fake)]  —  class 1 is P(fake).

    Training: FF++ pre-training (DeepfakeBench) + optional DLC fine-tuning.
    See scripts/finetune_pretrained.py and scripts/export_onnx.py.
    """

    model_note = _MODEL_NOTE

    def warmup(self) -> None:
        """Pre-load the model (call once at startup to avoid first-frame latency)."""
        ok = _load_session()
        if ok:
            # Run one dummy inference to warm up the graph
            dummy = np.zeros((1, 3, _INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
            try:
                _SESSION.run(None, {_INPUT_NAME: dummy})  # type: ignore[index]
                LOGGER.debug("CNN warmup OK")
            except Exception as exc:
                LOGGER.warning("CNN warmup inference failed: %s", exc)

    def score(self, frame: np.ndarray,
              face_bbox: tuple[int, int, int, int]) -> float | None:
        """Return P(fake) for the face detected in *frame* at *face_bbox*.

        Args:
            frame:     BGR frame from OpenCV.
            face_bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.

        Returns:
            float in [0, 1], or None if inference could not be run.
        """
        if not _load_session():
            return None

        try:
            face_bgr = crop_face(frame, face_bbox, margin=0.2)
            if face_bgr is None or face_bgr.size == 0:
                return None

            tensor = _preprocess(face_bgr)                      # (1,3,380,380)
            logits = _SESSION.run(None, {_INPUT_NAME: tensor})[0]  # (1,2) or (1,1)

            if _N_OUTPUTS == 2:
                # 2-class softmax: index 1 = P(fake)
                exp   = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probs = exp / exp.sum(axis=-1, keepdims=True)
                return float(np.clip(probs[0, 1], 0.0, 1.0))
            else:
                # 1-class sigmoid (timm train.py output)
                return float(np.clip(1.0 / (1.0 + np.exp(-logits[0, 0])), 0.0, 1.0))

        except Exception as exc:
            LOGGER.exception("CNN detector failed: %s", exc)
            return None

