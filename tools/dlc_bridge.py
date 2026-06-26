#!/usr/bin/env python3
"""dlc_bridge.py — Live toggle demo: Deep-Live-Cam feed → deepfake-detector.

Three-thread pipeline so display never blocks on slow inference:
  Thread 1 (SwapWorker)   — DLC face-swap via CoreML, ~15-20 fps
  Thread 2 (DetectWorker) — deepfake detector, runs every DETECT_EVERY frames
  Main thread             — display at camera FPS, overlays last known result

Usage
-----
    python3.11 tools/dlc_bridge.py \\
        --source /path/to/marquez.jpg \\
        --dlc    ~/PROJECTS/Deep-Live-Cam \\
        --camera 0

Controls
--------
    SPACE   toggle deepfake swap on / off
    Q / ESC quit
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


# ── tunables ──────────────────────────────────────────────────────────────────
DETECT_EVERY = 5          # run full detector pipeline every N swapped frames
DISPLAY_H    = 360        # panel height in pixels
QUEUE_MAX    = 2          # max frames buffered between threads

# ── colours ───────────────────────────────────────────────────────────────────
_GREEN  = (0, 220, 80)
_RED    = (0, 60, 240)
_GREY   = (140, 140, 140)
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)
_YELLOW = (0, 215, 255)


# ── overlay helpers ───────────────────────────────────────────────────────────

def _bar(frame, x, y, w, h, value, colour):
    cv2.rectangle(frame, (x, y), (x + w, y + h), _GREY, 1)
    filled = int(w * max(0.0, min(1.0, value)))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), colour, -1)


def _draw_detection_overlay(frame, result, swap_on):
    if result is None:
        cv2.putText(frame, "ANALYSING…", (10, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, _YELLOW, 1, cv2.LINE_AA)
        return

    conf       = result.confidence
    is_fake    = result.is_fake
    vc         = _RED if is_fake else _GREEN
    verdict    = f"{'FAKE' if is_fake else 'REAL'}  {conf:.2f}"

    pad, panel_w, panel_h = 10, 230, 112
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, verdict, (pad + 8, pad + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, vc, 2, cv2.LINE_AA)
    _bar(frame, pad + 8, pad + 34, panel_w - 20, 8, conf, vc)

    s = result.signals or {}
    for i, (name, key) in enumerate([("CNN", "cnn"), ("Temporal", "temporal"), ("Liveness", "liveness")]):
        score = s.get(key)
        yb    = pad + 58 + i * 22
        cv2.putText(frame, f"{name}:", (pad + 8, yb),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
        if score is not None:
            _bar(frame, pad + 80, yb - 10, 80, 7, score, vc)
            cv2.putText(frame, f"{score:.2f}", (pad + 168, yb),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "wait…", (pad + 80, yb),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)


def _draw_status_bar(canvas, swap_on, fps, detect_ms):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - 28), (w, h), (30, 30, 30), -1)
    sc = _GREEN if swap_on else _GREY
    cv2.putText(canvas, "SWAP: ON" if swap_on else "SWAP: OFF",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[SPACE] toggle  [Q] quit",
                (120, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)
    info = f"FPS: {fps:.1f}  det: {detect_ms:.0f}ms"
    cv2.putText(canvas, info, (w - 170, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)


# ── DLC swap (runs in its own thread) ─────────────────────────────────────────

class _DLCSwapper:
    """Wraps DLC face-swap; uses CoreML when available."""

    def __init__(self, dlc_path: str, source_img_path: str) -> None:
        sys.path.insert(0, dlc_path)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        import types as _types
        _ui_stub = _types.ModuleType("modules.ui")
        _ui_stub.update_status = lambda *a, **kw: None
        _ui_stub.check_and_ignore_nsfw = lambda *a, **kw: False
        sys.modules.setdefault("modules.ui", _ui_stub)

        import onnxruntime as ort
        providers = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                     if "CoreMLExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])

        import modules.globals as g
        g.execution_providers  = providers
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

        print(f"[dlc] Loading face analyser (providers: {providers})…")
        from modules.face_analyser import get_face_analyser, get_one_face
        from modules import imread_unicode
        get_face_analyser()

        print("[dlc] Loading face swapper model…")
        from modules.processors.frame.core import get_frame_processors_modules
        fps = get_frame_processors_modules(["face_swapper"])
        self._fp           = next(fp for fp in fps if fp.NAME == "DLC.FACE-SWAPPER")
        self._detect_one   = None
        from modules.face_analyser import detect_one_face_fast
        self._detect_one   = detect_one_face_fast

        src                = imread_unicode(source_img_path)
        self._source_face  = get_one_face(src)
        if self._source_face is None:
            raise ValueError(f"No face found in source image: {source_img_path}")
        print(f"[dlc] Source face loaded ✓  ({Path(source_img_path).name})")

    def swap(self, frame: np.ndarray) -> np.ndarray:
        target_face = self._detect_one(frame)
        if target_face is None:
            return frame
        out    = self._fp.swap_face(self._source_face, target_face, frame.copy())
        bboxes = []
        if hasattr(target_face, "bbox") and target_face.bbox is not None:
            bboxes.append(target_face.bbox.astype(int))
        return self._fp.apply_post_processing(out, bboxes)


class _SwapWorker(threading.Thread):
    """Reads raw frames, applies swap (or passthrough), pushes to out_q."""

    def __init__(self, swapper: _DLCSwapper,
                 in_q: queue.Queue, out_q: queue.Queue,
                 swap_flag: threading.Event, stop_flag: threading.Event):
        super().__init__(daemon=True)
        self._swapper   = swapper
        self._in_q      = in_q
        self._out_q     = out_q
        self._swap_flag = swap_flag
        self._stop_flag = stop_flag

    def run(self):
        while not self._stop_flag.is_set():
            try:
                raw = self._in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if self._swap_flag.is_set():
                try:
                    frame = self._swapper.swap(raw)
                except Exception:
                    frame = raw
            else:
                frame = raw
            # drop oldest if consumer is slow
            try:
                self._out_q.put_nowait((raw, frame))
            except queue.Full:
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._out_q.put_nowait((raw, frame))
                except queue.Full:
                    pass


class _DetectWorker(threading.Thread):
    """Runs the deepfake detector every DETECT_EVERY frames, stores last result."""

    def __init__(self, detector,
                 in_q: queue.Queue, stop_flag: threading.Event):
        super().__init__(daemon=True)
        self._detector   = detector
        self._in_q       = in_q
        self._stop_flag  = stop_flag
        self.result      = None
        self.detect_ms   = 0.0
        self._lock       = threading.Lock()
        self._frame_n    = 0

    def run(self):
        while not self._stop_flag.is_set():
            try:
                frame = self._in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            self._frame_n += 1
            if self._frame_n % DETECT_EVERY != 0:
                continue
            t0 = time.perf_counter()
            try:
                r = self._detector.analyse(frame)
            except Exception:
                continue
            ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self.result    = r
                self.detect_ms = ms
            # verbose terminal output to diagnose signal quality
            s = r.signals or {}
            print(
                f"[detect] {'FAKE' if r.is_fake else 'REAL'} {r.confidence:.2f} | "
                f"cnn={s.get('cnn', 'n/a')!r:.5} "
                f"temporal={s.get('temporal', 'n/a')!r:.5} "
                f"liveness={s.get('liveness', 'n/a')!r:.5} "
                f"({ms:.0f}ms)"
            )

    def get_result(self):
        with self._lock:
            return self.result, self.detect_ms


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live DLC → deepfake-detector bridge with toggle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Path to the source face image (e.g. marquez.jpg)")
    parser.add_argument("--dlc",
                        default=str(Path.home() / "PROJECTS" / "Deep-Live-Cam"),
                        help="Path to the Deep-Live-Cam repository root")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam device index")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-detector", action="store_true",
                        help="Skip deepfake detector (swap preview only)")
    args = parser.parse_args()

    if not Path(args.source).is_file():
        print(f"ERROR: source image not found: {args.source}", file=sys.stderr)
        return 1
    if not Path(args.dlc).is_dir():
        print(f"ERROR: DLC path not found: {args.dlc}", file=sys.stderr)
        return 1

    # ── detector ────────────────────────────────────────────────────────────
    detector = None
    if not args.no_detector:
        _repo = Path(__file__).resolve().parent.parent
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        try:
            print("[detector] Loading deepfake detector…")
            from deepfake_detector import Detector
            detector = Detector()
            print("[detector] Ready ✓")
        except Exception as exc:
            print(f"[detector] Could not load ({exc}); swap-only mode.")

    # ── DLC swapper ──────────────────────────────────────────────────────────
    try:
        swapper = _DLCSwapper(args.dlc, args.source)
    except Exception as exc:
        print(f"ERROR initialising DLC swapper: {exc}", file=sys.stderr)
        return 1

    # ── camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        return 1

    # ── thread plumbing ──────────────────────────────────────────────────────
    swap_flag  = threading.Event()   # set = deepfake ON
    stop_flag  = threading.Event()

    raw_q      = queue.Queue(maxsize=QUEUE_MAX)   # cam  → swap worker
    swapped_q  = queue.Queue(maxsize=QUEUE_MAX)   # swap → display + detect
    detect_q   = queue.Queue(maxsize=QUEUE_MAX)   # display → detect worker

    swap_worker = _SwapWorker(swapper, raw_q, swapped_q, swap_flag, stop_flag)
    swap_worker.start()

    detect_worker = None
    if detector is not None:
        detect_worker = _DetectWorker(detector, detect_q, stop_flag)
        detect_worker.start()

    print("\n── DLC Bridge running ──────────────────────────────────────────")
    print("  SPACE  toggle deepfake on/off")
    print("  Q/ESC  quit")
    print("────────────────────────────────────────────────────────────────\n")

    frame_times: list[float] = []

    try:
        while True:
            t0 = time.perf_counter()

            ret, raw_frame = cap.read()
            if not ret:
                break

            # push raw frame to swap worker (non-blocking, drop if full)
            try:
                raw_q.put_nowait(raw_frame)
            except queue.Full:
                pass

            # grab latest swapped pair (non-blocking, use last if empty)
            raw_display = raw_frame
            out_frame   = raw_frame
            try:
                raw_display, out_frame = swapped_q.get_nowait()
            except queue.Empty:
                pass

            # push output to detect worker (non-blocking)
            if detect_worker is not None:
                try:
                    detect_q.put_nowait(out_frame)
                except queue.Full:
                    pass

            # ── build display ────────────────────────────────────────────────
            out_display = out_frame.copy()
            if detect_worker is not None:
                result, detect_ms = detect_worker.get_result()
                _draw_detection_overlay(out_display, result, swap_flag.is_set())
            else:
                detect_ms = 0.0

            ph    = DISPLAY_H
            scale = ph / raw_display.shape[0]
            pw    = int(raw_display.shape[1] * scale)
            left  = cv2.resize(raw_display, (pw, ph))
            right = cv2.resize(out_display, (pw, ph))

            cv2.putText(left, "RAW", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREY, 1, cv2.LINE_AA)
            swap_on = swap_flag.is_set()
            cv2.putText(right,
                        "DEEPFAKE (ON)" if swap_on else "REAL (swap off)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        _RED if swap_on else _GREEN, 1, cv2.LINE_AA)

            divider = np.full((ph, 4, 3), 60, dtype=np.uint8)
            canvas  = np.hstack([left, divider, right])

            t1 = time.perf_counter()
            frame_times.append(t1 - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

            bar = np.zeros((28, canvas.shape[1], 3), dtype=np.uint8)
            canvas = np.vstack([canvas, bar])
            _draw_status_bar(canvas, swap_on, fps, detect_ms)

            cv2.imshow("DLC Bridge — deepfake-detector", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord(" "):
                if swap_flag.is_set():
                    swap_flag.clear()
                    label = "OFF ■  (real feed)"
                else:
                    swap_flag.set()
                    label = "ON ▶  (deepfake active)"

                # Flush stale frames from both queues so the swap worker
                # starts producing frames in the new state immediately.
                for q in (raw_q, swapped_q, detect_q):
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break

                # Reset detector temporal/liveness buffers so stale
                # pre-toggle signal history doesn't bleed into new state.
                if detector is not None:
                    detector.temporal_buffer.clear()
                    detector.liveness_analyser.reset()
                    if detect_worker is not None:
                        with detect_worker._lock:
                            detect_worker.result = None

                print(f"[bridge] Swap {label}")

    finally:
        stop_flag.set()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

