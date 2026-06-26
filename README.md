# deepfake-detector

**Real-time deepfake detection for video calls, live streams, and recorded media.**

Built by the [Alphinium](https://alphinium.com) project · Open source · Apache 2.0

[![CI](https://github.com/redsitesoftware/deepfake-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/redsitesoftware/deepfake-detector/actions)
[![PyPI](https://img.shields.io/pypi/v/deepfake-detector)](https://pypi.org/project/deepfake-detector/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## What this is

`deepfake-detector` is a runnable MVP with three active detection layers:

| Layer | What it detects | Implementation |
|-------|------------------|----------------|
| **CNN** | Frame-level artifacts in a face crop | HuggingFace ViT bootstrap model `dima806/deepfake_vs_real_image_detection` |
| **Temporal** | Frame-to-frame inconsistency, identity drift, flicker | Sliding window OpenCV analysis |
| **Liveness** | Blink patterns and head pose variance | MediaPipe FaceMesh + solvePnP |

The CNN model is intentionally marked as a bootstrap component and can be replaced later with a trained EfficientNet-B4 model.

---

## Quickstart

```bash
pip install -e ".[dev]"
```

```python
import cv2
from deepfake_detector import detect_frame

frame = cv2.imread("frame.jpg")
result = detect_frame(frame)

print(result.is_fake)
print(result.confidence)
print(result.signals)
```

### Run the API

```bash
uvicorn api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/detect/image -F "file=@photo.jpg"
```

### Run the demo

```bash
python demo.py --source 0
python demo.py --source sample.mp4 --output annotated.mp4
```

---

## Architecture

```
Input frame / stream
        │
        ├─► Face detection (OpenCV Haar cascade)
        │         ├─► CNN layer       HuggingFace ViT bootstrap detector
        │         ├─► Temporal layer  Flicker + histogram drift scoring
        │         └─► Liveness layer  EAR blink analysis + head pose variance
        │
        └─► Orchestrator → DetectionResult / FastAPI response
```

Models are lazy-loaded. Importing `deepfake_detector` is fast, and the HuggingFace model download happens on first use of the CNN detector or FastAPI startup warm-up.

---

## API surface

- `detect_frame(frame) -> DetectionResult`
- `detect_stream(source, callback, fps_limit=10) -> None`
- `detect_file(path) -> list[DetectionResult]`
- `POST /detect/image`
- `POST /detect/video`
- `WS /detect/stream`
- `GET /health`

---

## Development

```bash
pip install -e ".[dev]"
ruff check deepfake_detector api
pytest tests/ -v
```
