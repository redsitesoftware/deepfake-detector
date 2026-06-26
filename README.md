# deepfake-detector

**Real-time deepfake detection for video calls, live streams, and recorded media.**

Built by the [Alphinium](https://alphinium.com) project · Open source · Apache 2.0

[![CI](https://github.com/redsitesoftware/deepfake-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/redsitesoftware/deepfake-detector/actions)
[![PyPI](https://img.shields.io/pypi/v/deepfake-detector)](https://pypi.org/project/deepfake-detector/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## The problem

Deepfake attacks now occur every **5 minutes**. In 2024 alone:

- A finance employee wired **$25 million** after a multi-person deepfake Zoom call cloned his CFO and two executives (Arup, Hong Kong)
- A cybersecurity firm hired a **North Korean state actor** who passed 4 video interviews using an AI-enhanced photo
- A US Senator received a deepfake Zoom call impersonating a foreign minister — prompting a Senate-wide security alert

The tools to create these attacks are free, open source, and require a single photo. The tools to detect them — until now — were academic frameworks with broken download links and no production API.

---

## What this is

`deepfake-detector` is a Python library and FastAPI service with four detection layers:

| Layer | What it detects | Signal |
|-------|----------------|--------|
| **CNN** | Frame-level artifacts — blending boundaries, frequency anomalies | EfficientNet-B4, cross-domain AUC 99.44% |
| **Temporal** | Frame-to-frame inconsistency, identity drift, flicker | Sliding window analysis — gap no other OSS tool fills |
| **Liveness** | Eye blink patterns, head pose variance, gaze consistency | Catches injection attacks via virtual cameras |
| **Audio** | Voice synthesis, vocal tract anomalies | Whisper-feature classifier |

---

## Quickstart

```bash
pip install deepfake-detector
```

```python
import cv2
from deepfake_detector import detect_frame

frame = cv2.imread("frame.jpg")
result = detect_frame(frame)

print(result.is_fake)        # True / False
print(result.confidence)     # 0.0 – 1.0
print(result.signals)        # {'cnn': 0.91, 'temporal': 0.78, 'liveness': 0.85, 'audio': None}
```

### Run the API

```bash
docker run -p 8000:8000 redsitesoftware/deepfake-detector
```

```bash
# Single image
curl -X POST http://localhost:8000/detect/image \
  -F "file=@photo.jpg"

# Live stream (WebSocket)
# Connect to ws://localhost:8000/detect/stream
# Send: { "frame": "<base64 jpeg>", "frame_id": 1 }
# Receive: { "frame_id": 1, "is_fake": true, "confidence": 0.87, "signals": {...}, "latency_ms": 42 }
```

---

## Architecture

```
Input (image / video / live stream / audio)
        │
        ├─► Face Detection (MTCNN)
        │         │
        │         ├─► CNN Layer          EfficientNet-B4 · ONNX · fp16/fp32
        │         ├─► Temporal Layer     Sliding window · identity drift · flicker
        │         └─► Liveness Layer     MediaPipe · EAR blink · head pose variance
        │
        ├─► Audio Layer                  Whisper encoder · ASVspoof classifier
        │
        └─► Orchestrator → DetectionResult
```

Models are downloaded automatically from HuggingFace Hub on first use. No manual setup.

---

## API reference

### `detect_frame(frame, model_version='latest') → DetectionResult`

Analyse a single image frame.

### `detect_stream(source, callback, fps_limit=10) → None`

Process a live video stream. `source` is a webcam index, RTSP URL, or file path. Calls `callback(result)` for each analysed frame.

### `detect_file(path) → VideoDetectionResponse`

Analyse a recorded video file. Returns per-frame results and an aggregate verdict.

### `DetectionResult`

```python
@dataclass
class DetectionResult:
    is_fake: bool
    confidence: float           # 0.0–1.0
    signals: dict               # per-layer scores
    face_detected: bool
    face_bbox: tuple | None     # (x, y, w, h)
    latency_ms: float
    model_version: str
```

---

## REST API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/detect/image` | Single image upload |
| `POST` | `/detect/video` | Video file upload |
| `WS`   | `/detect/stream` | Live frame stream |
| `GET`  | `/health` | Service health + model versions |
| `GET`  | `/metrics` | Prometheus metrics |

---

## Detection thresholds

| Confidence | Suggested action |
|-----------|-----------------|
| < 0.3 | Real — log only |
| 0.3 – 0.7 | Uncertain — flag for review |
| > 0.7 | Fake — alert |
| > 0.9 | High confidence fake |

See [BENCHMARKS.md](BENCHMARKS.md) for FAR/FRR at each threshold.

---

## Limitations

This is not a silver bullet. Known evasion techniques exist:

- **Noise injection** — adding calibrated noise to deepfake video degrades CNN performance
- **Compression** — heavy H.264 re-encoding destroys high-frequency artifacts
- **Temporal smoothing** — low-pass filtering synthetic video reduces flicker signal

We document these honestly. See [LIMITATIONS.md](LIMITATIONS.md).

---

## Roadmap

| Milestone | Status | Issues |
|-----------|--------|--------|
| **v0.1** — Core library + API | 🔨 In progress | [#1](../../issues/1) [#2](../../issues/2) [#5](../../issues/5) [#6](../../issues/6) |
| **v0.2** — Full detection pipeline | 📋 Planned | [#3](../../issues/3) [#4](../../issues/4) [#7](../../issues/7) [#8](../../issues/8) [#11](../../issues/11) [#12](../../issues/12) |
| **v1.0** — Production release | 📋 Planned | [#10](../../issues/10) [#14](../../issues/14) |

---

## Built by Alphinium

This project is built and maintained by [Alphinium](https://alphinium.com) — AI-powered software platform. It powers deepfake verification in [ChatInstance](https://chatinstance.com) video sessions.

**Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

> The detection models in this project are trained for defensive security research. Use responsibly and in accordance with applicable laws.
