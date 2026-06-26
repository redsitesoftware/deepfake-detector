#!/usr/bin/env python3
"""dlc_bridge.py — Live toggle demo: Deep-Live-Cam feed → deepfake-detector.

Runs the Deep-Live-Cam face-swap pipeline headlessly alongside the deepfake
detector. Press SPACE to toggle the swap on/off. The detector sees what you
toggle, so you can watch it flip between REAL and FAKE in real time.

Usage
-----
    # From the Deep-Live-Cam venv (which has insightface / onnxruntime):
    python3.11 tools/dlc_bridge.py \\
        --source /path/to/marquez.jpg \\
        --dlc    /Users/dan/PROJECTS/Deep-Live-Cam \\
        --camera 0

Controls
--------
    SPACE   toggle deepfake swap on / off
    Q / ESC quit

Window layout
-------------
    ┌─────────────────┬────────────────────────────────────┐
    │   RAW WEBCAM    │          OUTPUT FRAME              │
    │   (always real) │  (swap ON → deepfake, OFF → real)  │
    │                 │  ┌──────────────────────────────┐  │
    │                 │  │  FAKE  0.87 ████████░░       │  │
    │                 │  │  CNN:      0.91               │  │
    │                 │  │  Temporal: 0.82               │  │
    │                 │  │  Liveness: 0.74               │  │
    │                 │  └──────────────────────────────┘  │
    ├─────────────────┴────────────────────────────────────┤
    │  SWAP: ON [SPACE toggle]    FPS: 24.1                │
    └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── colours ──────────────────────────────────────────────────────────────────
_GREEN  = (0, 220, 80)
_RED    = (0, 60, 240)
_ORANGE = (0, 165, 255)
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)
_GREY   = (140, 140, 140)
_YELLOW = (0, 215, 255)


# ── overlay helpers ───────────────────────────────────────────────────────────

def _bar(frame: np.ndarray, x: int, y: int, w: int, h: int,
         value: float, colour: tuple) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), _GREY, 1)
    filled = int(w * max(0.0, min(1.0, value)))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), colour, -1)


def _draw_detection_overlay(frame: np.ndarray, result, swap_on: bool) -> np.ndarray:
    """Render detection result in the top-left corner of *frame* (in-place)."""
    h, w = frame.shape[:2]
    pad, lh = 10, 22

    if result is None:
        label = "ANALYSING…"
        cv2.putText(frame, label, (pad, pad + lh),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, _YELLOW, 1, cv2.LINE_AA)
        return frame

    conf = result.confidence
    is_fake = result.is_fake

    verdict_colour = _RED if is_fake else _GREEN
    verdict_label  = f"{'FAKE' if is_fake else 'REAL'}  {conf:.2f}"

    # semi-transparent background panel
    panel_w, panel_h = 230, 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # verdict line
    cv2.putText(frame, verdict_label, (pad + 8, pad + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, verdict_colour, 2, cv2.LINE_AA)
    _bar(frame, pad + 8, pad + 34, panel_w - 20, 8, conf, verdict_colour)

    # per-signal breakdown
    signals = [
        ("CNN",      result.cnn_score),
        ("Temporal", result.temporal_score),
        ("Liveness", result.liveness_score),
    ]
    for i, (name, score) in enumerate(signals):
        y_base = pad + 58 + i * lh
        score_str = f"{score:.2f}" if score is not None else "wait…"
        cv2.putText(frame, f"{name}:", (pad + 8, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
        if score is not None:
            _bar(frame, pad + 80, y_base - 10, 80, 7, score, verdict_colour)
            cv2.putText(frame, score_str, (pad + 168, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, score_str, (pad + 80, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)

    return frame


def _draw_status_bar(canvas: np.ndarray, swap_on: bool, fps: float) -> None:
    """Draw bottom status strip on the combined canvas."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - 28), (w, h), (30, 30, 30), -1)
    swap_colour = _GREEN if swap_on else _GREY
    swap_text = "SWAP: ON" if swap_on else "SWAP: OFF"
    cv2.putText(canvas, swap_text, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, swap_colour, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[SPACE] toggle  [Q] quit", (120, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (w - 90, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)


# ── DLC swap helper ───────────────────────────────────────────────────────────

class _DLCSwapper:
    """Thin wrapper around Deep-Live-Cam's face-swap pipeline."""

    def __init__(self, dlc_path: str, source_img_path: str) -> None:
        sys.path.insert(0, dlc_path)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # Stub modules.ui before any DLC import triggers it — face_swapper
        # → modules.core → modules.ui → PySide6 (not in bridge venv).
        import types as _types
        _ui_stub = _types.ModuleType("modules.ui")
        _ui_stub.update_status = lambda *a, **kw: None
        _ui_stub.check_and_ignore_nsfw = lambda *a, **kw: False
        sys.modules.setdefault("modules.ui", _ui_stub)

        import modules.globals as g
        g.execution_providers  = ["CPUExecutionProvider"]
        g.frame_processors     = ["face_swapper"]
        g.many_faces           = False
        g.map_faces            = False
        g.mouth_mask           = False
        g.opacity              = 1.0
        g.sharpness            = 0.0
        g.enable_interpolation = False
        g.fp_ui                = {
            "face_enhancer": False,
            "face_enhancer_gpen256": False,
            "face_enhancer_gpen512": False,
        }
        g.source_path = source_img_path

        print("[dlc] Loading face analyser…")
        from modules.face_analyser import get_face_analyser, get_one_face
        from modules import imread_unicode
        get_face_analyser()  # warm up

        print("[dlc] Loading face swapper model…")
        from modules.processors.frame.core import get_frame_processors_modules
        fps = get_frame_processors_modules(["face_swapper"])
        self._fp = next(fp for fp in fps if fp.NAME == "DLC.FACE-SWAPPER")

        src_img = imread_unicode(source_img_path)
        self._source_face = get_one_face(src_img)
        if self._source_face is None:
            raise ValueError(f"No face found in source image: {source_img_path}")
        print(f"[dlc] Source face loaded ✓  ({Path(source_img_path).name})")

        self._detect_one = None
        from modules.face_analyser import detect_one_face_fast
        self._detect_one = detect_one_face_fast

    def swap(self, frame: np.ndarray) -> np.ndarray:
        target_face = self._detect_one(frame)
        if target_face is None:
            return frame
        out = self._fp.swap_face(self._source_face, target_face, frame.copy())
        bboxes = []
        if hasattr(target_face, "bbox") and target_face.bbox is not None:
            bboxes.append(target_face.bbox.astype(int))
        return self._fp.apply_post_processing(out, bboxes)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live DLC → deepfake-detector bridge with toggle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to the source face image (e.g. marquez.jpg)",
    )
    parser.add_argument(
        "--dlc",
        default=str(Path.home() / "PROJECTS" / "Deep-Live-Cam"),
        help="Path to the Deep-Live-Cam repository root",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Capture width",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Capture height",
    )
    parser.add_argument(
        "--no-detector", action="store_true",
        help="Skip deepfake detector (swap preview only — faster)",
    )
    args = parser.parse_args()

    # ── validate paths ──────────────────────────────────────────────────────
    if not Path(args.source).is_file():
        print(f"ERROR: source image not found: {args.source}", file=sys.stderr)
        return 1
    if not Path(args.dlc).is_dir():
        print(f"ERROR: DLC path not found: {args.dlc}", file=sys.stderr)
        return 1

    # ── deepfake-detector (optional — graceful if deps missing) ─────────────
    detector = None
    if not args.no_detector:
        # Ensure the detector package is importable (works whether run from
        # inside the repo or from anywhere with the repo on PYTHONPATH).
        _repo = Path(__file__).resolve().parent.parent
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        try:
            print("[detector] Loading deepfake detector…")
            from deepfake_detector import Detector
            detector = Detector()
            print("[detector] Ready ✓")
        except Exception as exc:
            print(f"[detector] Could not load detector ({exc}); running swap-only mode.")

    # ── DLC swapper ──────────────────────────────────────────────────────────
    try:
        swapper = _DLCSwapper(args.dlc, args.source)
    except Exception as exc:
        print(f"ERROR initialising DLC swapper: {exc}", file=sys.stderr)
        return 1

    # ── camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.camera}", file=sys.stderr)
        return 1

    print("\n── DLC Bridge running ──────────────────────────────────────────")
    print("  SPACE  toggle deepfake on/off")
    print("  Q/ESC  quit")
    print("────────────────────────────────────────────────────────────────\n")

    swap_on       = False
    last_result   = None
    frame_times: list[float] = []

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        raw_display = frame.copy()

        # ── apply swap ───────────────────────────────────────────────────────
        if swap_on:
            try:
                output_frame = swapper.swap(frame)
            except Exception:
                output_frame = frame
        else:
            output_frame = frame

        # ── run detector ─────────────────────────────────────────────────────
        if detector is not None:
            try:
                last_result = detector.analyse(output_frame)
            except Exception:
                pass  # keep last_result on transient errors

        # ── build side-by-side display ───────────────────────────────────────
        out_display = output_frame.copy()
        if detector is not None:
            _draw_detection_overlay(out_display, last_result, swap_on)

        # resize both panels to same height
        ph = 360
        scale = ph / raw_display.shape[0]
        pw = int(raw_display.shape[1] * scale)
        left  = cv2.resize(raw_display, (pw, ph))
        right = cv2.resize(out_display, (pw, ph))

        # label panels
        cv2.putText(left,  "RAW",    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREY,  1, cv2.LINE_AA)
        swap_label = "DEEPFAKE (ON)" if swap_on else "REAL (swap off)"
        swap_lc    = _RED if swap_on else _GREEN
        cv2.putText(right, swap_label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, swap_lc, 1, cv2.LINE_AA)

        divider = np.full((ph, 4, 3), 60, dtype=np.uint8)
        canvas  = np.hstack([left, divider, right])

        # status bar (adds 28px to bottom)
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

        bar_strip = np.zeros((28, canvas.shape[1], 3), dtype=np.uint8)
        canvas = np.vstack([canvas, bar_strip])
        _draw_status_bar(canvas, swap_on, fps)

        cv2.imshow("DLC Bridge — deepfake-detector", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord(" "):
            swap_on = not swap_on
            print(f"[bridge] Swap {'ON ▶  (deepfake active)' if swap_on else 'OFF ■  (real feed)'}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
