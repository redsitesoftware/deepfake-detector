"""live_test.py — Integrated DLC face-swap + deepfake detection WebSocket.

Flow:
  1. POST /api/source          — upload a face image, receive session_id
  2. WS  /ws/live/{session_id} — stream webcam frames (base64 JPEG),
                                  receive swapped frame + detection scores
  3. DELETE /api/source/{id}   — clean up session (optional, auto on WS close)
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from deepfake_detector import Detector

logger = logging.getLogger(__name__)
router = APIRouter()

# ── DLC path ──────────────────────────────────────────────────────────────────
_DLC_PATH = Path(os.environ.get("DLC_PATH", Path.home() / "PROJECTS" / "Deep-Live-Cam"))
_DLC_LOADED = False
_DLC_LOCK = threading.Lock()


def _init_dlc() -> bool:
    global _DLC_LOADED
    if _DLC_LOADED:
        return True
    with _DLC_LOCK:
        if _DLC_LOADED:
            return True
        if not _DLC_PATH.exists():
            logger.warning("DLC path not found: %s — swap disabled", _DLC_PATH)
            return False
        try:
            if str(_DLC_PATH) not in sys.path:
                sys.path.insert(0, str(_DLC_PATH))

            import modules.globals as dlc_globals
            dlc_globals.execution_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            dlc_globals.many_faces = False
            dlc_globals.mouth_mask = False
            dlc_globals.poisson_blend = False
            dlc_globals.sharpness = 0.0
            dlc_globals.enable_interpolation = False
            dlc_globals.nsfw_filter = False
            _DLC_LOADED = True
            logger.info("DLC loaded from %s", _DLC_PATH)
            return True
        except Exception as exc:
            logger.error("DLC init failed: %s", exc)
            return False


def _get_source_face(img: np.ndarray) -> Any | None:
    try:
        from modules.face_analyser import get_one_face
        return get_one_face(img)
    except Exception as exc:
        logger.error("face_analyser error: %s", exc)
        return None


def _swap_frame(source_face: Any, frame: np.ndarray) -> np.ndarray:
    try:
        from modules.processors.frame.face_swapper import process_frame
        return process_frame(source_face, frame.copy())
    except Exception as exc:
        logger.warning("swap_frame error: %s", exc)
        return frame


# ── Session store ─────────────────────────────────────────────────────────────

@dataclass
class Session:
    session_id: str
    source_face: Any          # insightface Face object
    source_thumb: bytes       # JPEG bytes for preview


_sessions: dict[str, Session] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_frame(payload: str) -> np.ndarray:
    data = payload.split(",", 1)[1] if "," in payload else payload
    buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode frame")
    return frame


def _encode_frame(frame: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Cannot encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _make_thumb(img: np.ndarray, size: int = 120) -> bytes:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/api/source")
async def upload_source(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a face image. Returns session_id to use with /ws/live/{session_id}."""
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "Empty file")

    buf = np.frombuffer(payload, dtype=np.uint8)

    # Try IMREAD_UNCHANGED first to preserve alpha/palette, then convert to BGR
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "Cannot decode image — send JPEG or PNG")

    # Normalise to 3-channel BGR regardless of source format
    if img.ndim == 2:                        # greyscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                  # RGBA/BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Upscale tiny images so InsightFace can find the face
    h, w = img.shape[:2]
    if max(h, w) < 256:
        scale = 256 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    dlc_ok = _init_dlc()
    source_face = None
    warning = None
    if dlc_ok:
        source_face = _get_source_face(img)
        if source_face is None:
            # Don't hard-fail — let the user proceed, DLC will attempt detection
            # per-frame on the webcam feed.  Warn so they know the swap may not work.
            warning = "Face not detected in upload — try a clearer front-facing photo. Swap may not work."
            logger.warning("Source face not detected in uploaded image")

    session_id = uuid.uuid4().hex
    _sessions[session_id] = Session(
        session_id=session_id,
        source_face=source_face,
        source_thumb=_make_thumb(img),
    )
    resp: dict = {
        "session_id": session_id,
        "dlc_available": dlc_ok,
        "face_detected": source_face is not None,
    }
    if warning:
        resp["warning"] = warning
    return JSONResponse(resp)


@router.get("/api/source/{session_id}/thumb")
async def source_thumb(session_id: str) -> Response:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return Response(content=session.source_thumb, media_type="image/jpeg")


@router.delete("/api/source/{session_id}")
async def delete_source(session_id: str) -> JSONResponse:
    _sessions.pop(session_id, None)
    return JSONResponse({"deleted": session_id})


@router.websocket("/ws/live/{session_id}")
async def live_stream(websocket: WebSocket, session_id: str) -> None:
    """Stream webcam frames → DLC swap → detection. Returns scores + swapped frame."""
    await websocket.accept()

    session = _sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "detail": "Session not found"})
        await websocket.close()
        return

    await websocket.send_json({"type": "ready", "fps_limit": 15})

    loop     = asyncio.get_event_loop()
    detector = Detector(fps=15.0)
    try:
        while True:
            try:
                data = await websocket.receive_json()
                raw_frame = _decode_frame(data["frame"])

                # ── DLC face swap (run in thread — blocks ~100-500ms) ──────
                if session.source_face is not None:
                    swapped = await loop.run_in_executor(
                        None, _swap_frame, session.source_face, raw_frame
                    )
                else:
                    swapped = raw_frame

                # ── Detection on swapped frame (also CPU-bound) ────────────
                result = await loop.run_in_executor(
                    None, detector.analyse, swapped
                )

                # ── Build response ─────────────────────────────────────────
                swap_b64 = _encode_frame(swapped)

                await websocket.send_json({
                    "frame_id":     data.get("frame_id"),
                    "swapped":      swap_b64,
                    "is_fake":      result.is_fake,
                    "confidence":   round(result.confidence, 4),
                    "signals":      {k: (round(v, 4) if v is not None else None)
                                     for k, v in result.signals.items()},
                    "face_detected": result.face_detected,
                    "face_bbox":    result.face_bbox,
                    "latency_ms":   round(result.latency_ms, 1),
                })

            except WebSocketDisconnect:
                raise
            except Exception as exc:
                try:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    finally:
        detector.close()
        # Don't delete the session here — client may reconnect (e.g. after temporal reset).
        # Session is cleaned up on DELETE /api/source/{session_id} or "Change Face".
