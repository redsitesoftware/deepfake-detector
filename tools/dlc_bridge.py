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
import collections
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


def _draw_detection_overlay(frame, result, embed_score, warmup_progress, swap_on):
    pad, panel_w, panel_h = 10, 235, 130

    # Warmup phase (building reference window — no enroll needed)
    if warmup_progress < 1.0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + 50), _BLACK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        pct = int(warmup_progress * 100)
        cv2.putText(frame, f"Calibrating… {pct}%", (pad + 8, pad + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _YELLOW, 1, cv2.LINE_AA)
        _bar(frame, pad + 8, pad + 30, panel_w - 20, 8, warmup_progress, _YELLOW)
        return

    # Identity drift drives the verdict
    id_score = embed_score if embed_score is not None else 0.0
    s        = (result.signals or {}) if result else {}

    temporal = s.get("temporal")
    liveness = s.get("liveness")
    scores   = [(id_score, 0.60)]
    if temporal is not None: scores.append((temporal, 0.25))
    if liveness is not None: scores.append((liveness, 0.15))
    total_w  = sum(w for _, w in scores)
    conf     = sum(v * w for v, w in scores) / total_w
    is_fake  = conf > 0.50

    vc      = _RED if is_fake else _GREEN
    verdict = f"{'FAKE' if is_fake else 'REAL'}  {conf:.2f}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, verdict, (pad + 8, pad + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, vc, 2, cv2.LINE_AA)
    _bar(frame, pad + 8, pad + 34, panel_w - 20, 8, conf, vc)

    signals = [
        ("ID drift", embed_score),
        ("Temporal", temporal),
        ("Liveness", liveness),
    ]
    for i, (name, score) in enumerate(signals):
        yb = pad + 58 + i * 22
        cv2.putText(frame, f"{name}:", (pad + 8, yb),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
        if score is not None:
            _bar(frame, pad + 88, yb - 10, 80, 7, score, vc)
            cv2.putText(frame, f"{score:.2f}", (pad + 176, yb),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "wait…", (pad + 88, yb),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)


def _draw_status_bar(canvas, swap_on, fps, detect_ms, show_picker_hint=False):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - 28), (w, h), (30, 30, 30), -1)
    sc = _GREEN if swap_on else _GREY
    cv2.putText(canvas, "SWAP: ON" if swap_on else "SWAP: OFF",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 1, cv2.LINE_AA)
    hint = "[SPACE] toggle  [G] pick face  [Q] quit" if show_picker_hint else "[SPACE] toggle  [Q] quit"
    cv2.putText(canvas, hint,
                (120, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, _GREY, 1, cv2.LINE_AA)
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

    def load_source(self, source_img_path: str) -> bool:
        """Hot-swap the source face. Returns True on success."""
        from modules import imread_unicode
        from modules.face_analyser import get_one_face
        src  = imread_unicode(source_img_path)
        face = get_one_face(src)
        if face is None:
            print(f"[dlc] No face found in {Path(source_img_path).name}")
            return False
        self._source_face = face
        print(f"[dlc] Source face updated ✓  ({Path(source_img_path).name})")
        return True

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
            # Snapshot swap state BEFORE the (potentially slow) swap call.
            # If SPACE fires during the swap, the state will have changed and
            # we discard this output rather than showing a stale mixed frame.
            was_swapping = self._swap_flag.is_set()
            if was_swapping:
                try:
                    frame = self._swapper.swap(raw)
                except Exception:
                    frame = raw
            else:
                frame = raw
            # Discard if toggle happened while we were swapping
            if self._swap_flag.is_set() != was_swapping:
                continue
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


class _EmbeddingVarianceDetector:
    """Detect face swap without a reference image — works on any face.

    Two complementary signals:

    1. SPIKE detection (catches the transition moment):
       Track frame-to-frame cosine distances in a rolling baseline window.
       A face swap causes a sudden z-score spike (anomalously large jump).
       Score stays 'sticky' for STICKY_SECS after a spike, then decays.

    2. NEAREST-NEIGHBOUR drift (catches sustained identity change):
       Compare current embedding to the *closest* reference frame from
       DELAY_LOW..DELAY_HIGH frames ago (best-matching pose). This is
       robust to head movement: if you tilted your head, some reference
       frame had the same tilt — the NN search finds it.
       A different person has no close match at all → high drift score.

    Combining both avoids false positives from natural head movement (drift)
    while still catching swaps that happened > STICKY_SECS ago (NN).
    """
    REF_SIZE     = 90    # total history kept
    DELAY_LOW    = 15    # reference zone: start (frames ago)
    DELAY_HIGH   = 60    # reference zone: end (frames ago)
    SPIKE_WIN    = 30    # rolling window for spike baseline
    SPIKE_K      = 3.0   # std devs above baseline = spike
    STICKY_SECS  = 5.0   # seconds to hold FAKE after a spike

    def __init__(self, detect_one_face_fn):
        self._detect    = detect_one_face_fn
        self._history:  collections.deque[np.ndarray] = collections.deque(maxlen=self.REF_SIZE)
        self._dists:    collections.deque[float]       = collections.deque(maxlen=self.SPIKE_WIN)
        self._last_emb: np.ndarray | None = None
        self._last_spike_t: float = -999.0

    @property
    def warmup_progress(self) -> float:
        # Need DELAY_HIGH valid embeddings before NN drift is usable
        return min(1.0, len(self._history) / self.DELAY_HIGH)

    def reset(self):
        self._history.clear()
        self._dists.clear()
        self._last_emb    = None
        self._last_spike_t = -999.0

    def update(self, frame: np.ndarray) -> float | None:
        """Returns fake score 0–1, or None during warmup."""
        face = self._detect(frame)
        if face is None or not hasattr(face, "embedding") or face.embedding is None:
            return None

        emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return None
        emb /= norm

        # ── spike detection (consecutive distance) ────────────────────────────
        spike_score = 0.0
        if self._last_emb is not None:
            d = float(1.0 - np.dot(self._last_emb, emb))
            if len(self._dists) >= 10:
                mu = float(np.mean(self._dists))
                sd = float(np.std(self._dists))  + 1e-6
                z  = (d - mu) / sd
                if z > self.SPIKE_K:
                    self._last_spike_t = time.monotonic()
                    print(f"[embed] SPIKE z={z:.1f} d={d:.4f} mu={mu:.4f}")
            self._dists.append(d)
        self._last_emb = emb

        self._history.append(emb)

        # Sticky: score decays linearly from 1→0 over STICKY_SECS
        age = time.monotonic() - self._last_spike_t
        spike_score = float(max(0.0, 1.0 - age / self.STICKY_SECS))

        # ── NN drift detection (needs full warmup) ────────────────────────────
        if len(self._history) < self.DELAY_HIGH:
            # Still warming up: return spike-only score (or None)
            return spike_score if spike_score > 0.0 else None

        hist = list(self._history)
        n    = len(hist)
        ref  = hist[max(0, n - self.DELAY_HIGH) : n - self.DELAY_LOW]
        if len(ref) < 5:
            return spike_score if spike_score > 0.0 else None

        # Nearest-neighbour: find the most similar reference frame
        sims     = np.array([float(np.dot(r, emb)) for r in ref])
        best_sim = float(sims.max())
        # best_sim ~0.92-0.99 same person, ~0.20-0.60 different person
        # Sigmoid midpoint at 0.75 (conservative — real faces are very close to themselves)
        nn_score = float(1.0 / (1.0 + np.exp(20.0 * (best_sim - 0.75))))

        # Take the higher of spike or NN
        return float(np.clip(max(spike_score, nn_score), 0.0, 1.0))


class _DetectWorker(threading.Thread):
    """Runs deepfake detector + embedding identity check every DETECT_EVERY frames."""

    def __init__(self, detector, embed_detector: _EmbeddingVarianceDetector,
                 in_q: queue.Queue, stop_flag: threading.Event):
        super().__init__(daemon=True)
        self._detector       = detector
        self._embed          = embed_detector
        self._in_q           = in_q
        self._stop_flag      = stop_flag
        self.result          = None
        self.embed_score     = None
        self.detect_ms       = 0.0
        self._lock           = threading.Lock()
        self._frame_n        = 0

    def run(self):
        while not self._stop_flag.is_set():
            try:
                frame = self._in_q.get(timeout=0.05)
            except queue.Empty:
                continue

            # Embedding check every frame (it's fast — just a dot product after detect)
            embed_score = self._embed.update(frame)

            self._frame_n += 1
            if self._frame_n % DETECT_EVERY != 0:
                if embed_score is not None:
                    with self._lock:
                        self.embed_score = embed_score
                continue

            t0 = time.perf_counter()
            try:
                r = self._detector.analyse(frame)
            except Exception:
                continue
            ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self.result      = r
                self.embed_score = embed_score
                self.detect_ms   = ms
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
            return self.result, self.embed_score, self.detect_ms


# ── image picker grid ─────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scan_images(directory: str) -> list[Path]:
    return sorted(
        p for p in Path(directory).iterdir()
        if p.suffix.lower() in _IMG_EXTS
    )


def _show_picker(directory: str, current: str | None = None) -> str | None:
    """Show a clickable grid of images from *directory*.
    Returns the selected path string, or None if cancelled (ESC/Q).
    """
    images = _scan_images(directory)
    if not images:
        print(f"[picker] No images found in {directory}")
        return None

    COLS      = 4
    THUMB     = 160   # thumbnail size (square)
    PAD       = 8
    LABEL_H   = 20
    CELL      = THUMB + PAD * 2
    CELL_H    = THUMB + PAD * 2 + LABEL_H
    rows      = (len(images) + COLS - 1) // COLS
    win_w     = COLS * CELL
    win_h     = rows * CELL_H + 40   # +40 for header

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

    # header
    cv2.putText(canvas, "Select source face  [G/ESC = cancel]",
                (PAD, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)

    thumbs: list[np.ndarray] = []
    for img_path in images:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
            img = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        except Exception:
            img = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
        thumbs.append(img)

    selected: list[str | None] = [None]
    done = threading.Event()

    def _render(highlight: int = -1) -> np.ndarray:
        c = canvas.copy()
        for idx, (img_path, thumb) in enumerate(zip(images, thumbs)):
            col = idx % COLS
            row = idx // COLS
            x   = col * CELL + PAD
            y   = row * CELL_H + 40 + PAD
            c[y:y + THUMB, x:x + THUMB] = thumb
            is_current  = str(img_path) == current
            is_highlight = idx == highlight
            colour = _GREEN if is_current else (_YELLOW if is_highlight else _GREY)
            cv2.rectangle(c, (x - 2, y - 2), (x + THUMB + 2, y + THUMB + 2), colour, 2)
            label = img_path.stem[:18]
            cv2.putText(c, label, (x, y + THUMB + LABEL_H - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _WHITE, 1, cv2.LINE_AA)
        return c

    win = "Face Picker"
    cv2.namedWindow(win)
    hover_idx = [-1]

    def _mouse(event, mx, my, flags, param):
        if my < 40:
            hover_idx[0] = -1
            return
        col = mx // CELL
        row = (my - 40) // CELL_H
        idx = row * COLS + col
        if 0 <= idx < len(images):
            hover_idx[0] = idx
            if event == cv2.EVENT_LBUTTONDOWN:
                selected[0] = str(images[idx])
                done.set()
        else:
            hover_idx[0] = -1

    cv2.setMouseCallback(win, _mouse)

    while not done.is_set():
        cv2.imshow(win, _render(hover_idx[0]))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("g"), ord("G"), ord("q"), ord("Q"), 27):
            break

    cv2.destroyWindow(win)
    return selected[0]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live DLC → deepfake-detector bridge with toggle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=None,
                        help="Path to a single source face image")
    parser.add_argument("--source-dir", default=None,
                        help="Directory of face images — opens a picker grid (press G to reopen)")
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

    # ── resolve source image ─────────────────────────────────────────────────
    source_dir = args.source_dir
    if source_dir and not Path(source_dir).is_dir():
        print(f"ERROR: source-dir not found: {source_dir}", file=sys.stderr)
        return 1
    if not args.source and not source_dir:
        print("ERROR: provide --source or --source-dir", file=sys.stderr)
        return 1

    source_path: str | None = args.source
    if source_dir and not source_path:
        source_path = _show_picker(source_dir)
        if not source_path:
            print("No face selected — exiting.")
            return 0

    if not Path(args.dlc).is_dir():
        print(f"ERROR: DLC path not found: {args.dlc}", file=sys.stderr)
        return 1

    # ── detector ────────────────────────────────────────────────────────────
    detector     = None
    raw_detector = None
    if not args.no_detector:
        _repo = Path(__file__).resolve().parent.parent
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        try:
            print("[detector] Loading deepfake detector…")
            from deepfake_detector import Detector
            detector     = Detector()   # scores the output (swapped or real)
            raw_detector = Detector()   # always scores the raw camera feed
            print("[detector] Ready ✓")
        except Exception as exc:
            print(f"[detector] Could not load ({exc}); swap-only mode.")

    # ── DLC swapper ──────────────────────────────────────────────────────────
    try:
        swapper = _DLCSwapper(args.dlc, source_path)
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

    # ── embedding detectors (uses insightface already loaded by swapper) ─────
    # Must use get_one_face (full pipeline) — detect_one_face_fast skips the
    # recognition model and returns Face objects without .embedding.
    from modules.face_analyser import get_one_face as _get_one_face_full
    embed_out = _EmbeddingVarianceDetector(_get_one_face_full)  # scores output feed
    embed_raw = _EmbeddingVarianceDetector(_get_one_face_full)  # always scores raw feed

    # ── thread plumbing ──────────────────────────────────────────────────────
    swap_flag  = threading.Event()   # set = deepfake ON
    stop_flag  = threading.Event()

    raw_q      = queue.Queue(maxsize=QUEUE_MAX)   # cam  → swap worker
    swapped_q  = queue.Queue(maxsize=QUEUE_MAX)   # swap → display + detect
    detect_q   = queue.Queue(maxsize=QUEUE_MAX)   # output frames → detect worker
    raw_det_q  = queue.Queue(maxsize=QUEUE_MAX)   # raw frames → raw detect worker

    swap_worker = _SwapWorker(swapper, raw_q, swapped_q, swap_flag, stop_flag)
    swap_worker.start()

    detect_worker     = None
    raw_detect_worker = None
    if detector is not None:
        detect_worker     = _DetectWorker(detector,     embed_out, detect_q,  stop_flag)
        raw_detect_worker = _DetectWorker(raw_detector, embed_raw, raw_det_q, stop_flag)
        detect_worker.start()
        raw_detect_worker.start()
        print("[embed] Enrolling real face — keep swap OFF for first 30 frames…")

    print("\n── DLC Bridge running ──────────────────────────────────────────")
    print("  SPACE  toggle deepfake on/off")
    if source_dir:
        print("  G      open face picker grid")
    print("  Q/ESC  quit")
    print("────────────────────────────────────────────────────────────────\n")

    frame_times: list[float] = []
    last_out_frame:   np.ndarray | None = None   # held across empty-queue frames
    last_raw_display: np.ndarray | None = None

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

            # also push raw frame to the raw detector (always scores real feed)
            if raw_detect_worker is not None:
                try:
                    raw_det_q.put_nowait(raw_frame)
                except queue.Full:
                    pass

            # Grab the latest swapped pair.  If the queue is empty (worker
            # is mid-processing) we HOLD the last known frame — never fall
            # back to raw_frame which causes real↔fake alternation.
            try:
                last_raw_display, last_out_frame = swapped_q.get_nowait()
            except queue.Empty:
                pass   # keep last_out_frame / last_raw_display

            # Bootstrap: before the first frame arrives show the camera feed
            raw_display = last_raw_display if last_raw_display is not None else raw_frame
            out_frame   = last_out_frame   if last_out_frame   is not None else raw_frame

            # push output to detect worker (non-blocking)
            if detect_worker is not None:
                try:
                    detect_q.put_nowait(out_frame)
                except queue.Full:
                    pass

            # ── build display ────────────────────────────────────────────────
            # Left panel: raw feed + its own detection score
            raw_disp = raw_display.copy()
            if raw_detect_worker is not None:
                raw_result, raw_embed, _ = raw_detect_worker.get_result()
                _draw_detection_overlay(raw_disp, raw_result, raw_embed,
                                        embed_raw.warmup_progress, swap_on=False)

            # Right panel: output feed + output detection score
            out_display = out_frame.copy()
            swap_on = swap_flag.is_set()
            if detect_worker is not None:
                result, out_embed, detect_ms = detect_worker.get_result()
                _draw_detection_overlay(out_display, result, out_embed,
                                        embed_out.warmup_progress, swap_on)
            else:
                detect_ms = 0.0

            ph    = DISPLAY_H
            scale = ph / raw_display.shape[0]
            pw    = int(raw_display.shape[1] * scale)
            left  = cv2.resize(raw_disp,     (pw, ph))
            right = cv2.resize(out_display,  (pw, ph))

            cv2.putText(left, "YOU (raw)", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREY, 1, cv2.LINE_AA)
            cv2.putText(right,
                        "DEEPFAKE (ON)" if swap_on else "OUTPUT (swap off)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
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
            _draw_status_bar(canvas, swap_on, fps, detect_ms, show_picker_hint=bool(source_dir))

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

                # Snap the held display frame to the current raw camera
                # image so we instantly show the correct state while
                # waiting for the first worker frame in the new state.
                last_out_frame   = raw_frame.copy()
                last_raw_display = raw_frame.copy()

                # Reset output detector state on toggle (fresh baseline for new mode)
                if detector is not None:
                    detector.temporal_buffer.clear()
                    detector.liveness_analyser.reset()
                    embed_out.reset()
                    if detect_worker is not None:
                        with detect_worker._lock:
                            detect_worker.result = None
                            detect_worker.embed_score = None
                # raw detector keeps its state — always sees the real face

                print(f"[bridge] Swap {label}")

            elif source_dir and key in (ord("g"), ord("G")):
                was_swapping = swap_flag.is_set()
                swap_flag.clear()
                new_path = _show_picker(source_dir, current=source_path)
                if new_path and new_path != source_path:
                    source_path = new_path
                    swapper.load_source(source_path)
                    for q in (raw_q, swapped_q, detect_q):
                        while not q.empty():
                            try: q.get_nowait()
                            except queue.Empty: break
                    last_out_frame = last_raw_display = None
                    # re-enroll since source face changed
                    embed_out.reset()
                    embed_raw.reset()
                    print("[embed] Re-enrolling — keep swap OFF for 30 frames…")
                if was_swapping:
                    swap_flag.set()

    finally:
        stop_flag.set()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

