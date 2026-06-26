from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from deepfake_detector import Detector

router = APIRouter()


def decode_base64_frame(payload: str) -> np.ndarray:
    encoded = payload.split(",", 1)[1] if "," in payload else payload
    buffer = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode frame")
    return frame


@router.websocket("/detect/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "ready", "fps_limit": 15})
    detector = Detector(fps=15.0)

    try:
        while True:
            try:
                data = await websocket.receive_json()
                frame = decode_base64_frame(data["frame"])
                result = detector.analyse(frame)
                await websocket.send_json(
                    {
                        "frame_id": data.get("frame_id"),
                        "is_fake": result.is_fake,
                        "confidence": result.confidence,
                        "signals": result.signals,
                        "face_detected": result.face_detected,
                        "latency_ms": result.latency_ms,
                    }
                )
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                try:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                except Exception:
                    break
                continue
    except WebSocketDisconnect:
        pass
    finally:
        detector.close()
