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
    pad, panel_w, panel_h = 10, 240, 110

    # Warmup phase
    if warmup_progress < 1.0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + 50), _BLACK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        pct = int(warmup_progress * 100)
        cv2.putText(frame, f"Calibrating… {pct}%", (pad + 8, pad + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _YELLOW, 1, cv2.LINE_AA)
        _bar(frame, pad + 8, pad + 30, panel_w - 20, 8, warmup_progress, _YELLOW)
        return

    # Embed score is the sole decider when available; fall back to temporal only
    # if face wasn't detected (embed_score is None).
    if embed_score is not None:
        conf    = embed_score
        src_lbl = "embed"
    else:
        s       = (result.signals or {}) if result else {}
        conf    = s.get("temporal", 0.0) or 0.0
        src_lbl = "temporal"

    is_fake = conf > 0.50
    vc      = _RED if is_fake else _GREEN
    verdict = f"{'FAKE' if is_fake else 'REAL'}  {conf:.2f}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, verdict, (pad + 8, pad + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, vc, 2, cv2.LINE_AA)
    _bar(frame, pad + 8, pad + 34, panel_w - 20, 8, conf, vc)

    cv2.putText(frame, f"via {src_lbl}", (pad + 8, pad + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GREY, 1, cv2.LINE_AA)

    # Sub-signals for debugging
    signals = [("Spike", None), ("NN drift", None)]  # filled via embed internals
    if embed_score is not None:
        pass  # composite score already shown
    else:
        s = (result.signals or {}) if result else {}
        for i, (name, key) in enumerate([("Temporal", "temporal"), ("Liveness", "liveness")]):
            score = s.get(key)
            yb    = pad + 72 + i * 20
            cv2.putText(frame, f"{name}:", (pad + 8, yb),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, _GREY, 1, cv2.LINE_AA)
            if score is not None:
                _bar(frame, pad + 88, yb - 9, 70, 6, score, _GREY)
                cv2.putText(frame, f"{score:.2f}", (pad + 165, yb),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, _GREY, 1, cv2.LINE_AA)


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
    """Detect face swap via nearest-neighbour identity drift (no reference needed).

    Strategy:
      Keep a rolling history of HISTORY_SIZE normalised face embeddings.
      Compare the current embedding against the *closest* match in the
      reference zone (DELAY_LOW..DELAY_HIGH frames ago).

      Pose-robustness: because the NN search scans all reference poses,
      if your current head angle appeared in the reference, it matches.
      A different person (face swap) has NO close match at all.

    Adaptive threshold:
      Rather than a hardcoded cutoff, the threshold is learned from the
      distribution of observed NN similarities (mean − 2.5σ). This adapts
      to the actual face: if you regularly get best_sim=0.92, threshold
      becomes ~0.87; a swap dropping to 0.35 is well below that.

    History is NOT cleared on toggle: the reference zone spans your real
    face frames. When swap turns ON, the current Marquez embedding diverges
    from those real-face references → FAKE detected within DELAY_LOW frames.

    Score is smoothed over SMOOTH_N frames to prevent rapid oscillation.
    Terminal logs every ~2s: best_sim, adaptive threshold, and verdict.
    """
    HISTORY_SIZE = 120
    DELAY_LOW    = 10    # reference zone start (frames ago)
    DELAY_HIGH   = 80    # reference zone end — wider zone = more pose variety captured
    SMOOTH_N     = 8     # rolling average to suppress oscillation

    def __init__(self, detect_one_face_fn):
        self._detect    = detect_one_face_fn
        self._history:  collections.deque[np.ndarray] = collections.deque(maxlen=self.HISTORY_SIZE)
        self._sim_hist: collections.deque[float]       = collections.deque(maxlen=300)
        self._scores:   collections.deque[float]       = collections.deque(maxlen=self.SMOOTH_N)
        self._log_t:    float = 0.0

    @property
    def warmup_progress(self) -> float:
        return min(1.0, len(self._history) / self.DELAY_HIGH)

    def reset(self):
        """Only call this when the source identity changes (face picker). 
        Do NOT call on swap toggle — we need pre-toggle frames as reference."""
        self._history.clear()
        self._sim_hist.clear()
        self._scores.clear()

    def _adaptive_thresh(self) -> float:
        """Learned threshold: mean − 2.5σ.  Needs ≥30 samples; floor at 0.45."""
        if len(self._sim_hist) < 30:
            return 0.60   # conservative fallback — avoids false FAKEs early on
        mu = float(np.mean(self._sim_hist))
        sd = float(np.std(self._sim_hist))
        return float(max(0.45, mu - 2.5 * sd))

    def update(self, frame: np.ndarray) -> float | None:
        """Returns smoothed fake score 0–1, or None during warmup."""
        face = self._detect(frame)
        if face is None or not hasattr(face, "embedding") or face.embedding is None:
            return None

        emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return None
        emb /= norm
        self._history.append(emb)

        if len(self._history) < self.DELAY_HIGH:
            return None   # still warming up

        hist = list(self._history)
        n    = len(hist)
        ref  = hist[max(0, n - self.DELAY_HIGH) : n - self.DELAY_LOW]
        if len(ref) < 5:
            return None

        # Nearest-neighbour: best matching pose in reference window
        best_sim = float(max(np.dot(r, emb) for r in ref))
        thresh   = self._adaptive_thresh()

        # Only add to threshold learner when it looks like the real/stable face
        # (don't let a sustained swap corrupt the baseline)
        if best_sim > thresh - 0.05:
            self._sim_hist.append(best_sim)

        # Sigmoid: best_sim >> thresh → score 0 (REAL); best_sim << thresh → score 1 (FAKE)
        raw_score = float(1.0 / (1.0 + np.exp(25.0 * (best_sim - thresh))))

        # Smooth over last SMOOTH_N frames to suppress oscillation
        self._scores.append(raw_score)
        score = float(np.mean(self._scores))

        # Terminal log every ~2s
        now = time.monotonic()
        if now - self._log_t > 2.0:
            self._log_t = now
            verdict = "FAKE" if score > 0.50 else "REAL"
            n_samples = len(self._sim_hist)
            print(f"[embed] {verdict:4s}  best_sim={best_sim:.3f}  "
                  f"thresh={thresh:.3f} ({n_samples} samples)  score={score:.2f}")

        return float(np.clip(score, 0.0, 1.0))


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

    # ── thread plumbing ──────────────────────────────────────────────────────
    swap_flag  = threading.Event()   # set = deepfake ON
    stop_flag  = threading.Event()

    raw_q      = queue.Queue(maxsize=QUEUE_MAX)   # cam  → swap worker
    swapped_q  = queue.Queue(maxsize=QUEUE_MAX)   # swap → display + detect
    detect_q   = queue.Queue(maxsize=QUEUE_MAX)   # output frames → detect worker

    swap_worker = _SwapWorker(swapper, raw_q, swapped_q, swap_flag, stop_flag)
    swap_worker.start()

    detect_worker = None
    if detector is not None:
        detect_worker = _DetectWorker(detector, embed_out, detect_q, stop_flag)
        detect_worker.start()
        print("[embed] Calibrating — keep swap OFF for first few seconds…")

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
            out_display = out_frame.copy()
            swap_on = swap_flag.is_set()
            if detect_worker is not None:
                result, out_embed, detect_ms = detect_worker.get_result()
                _draw_detection_overlay(out_display, result, out_embed,
                                        embed_out.warmup_progress, swap_on)
            else:
                detect_ms = 0.0

            ph    = DISPLAY_H
            scale = ph / out_frame.shape[0]
            pw    = int(out_frame.shape[1] * scale)
            panel = cv2.resize(out_display, (pw, ph))

            cv2.putText(panel,
                        "DEEPFAKE ON" if swap_on else "SWAP OFF",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        _RED if swap_on else _GREEN, 1, cv2.LINE_AA)

            canvas = panel

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

                # Do NOT reset embed_out on toggle — the reference window must
                # keep real-face frames so the NN can detect divergence when swap turns ON.
                if detector is not None:
                    detector.temporal_buffer.clear()
                    detector.liveness_analyser.reset()
                    if detect_worker is not None:
                        with detect_worker._lock:
                            detect_worker.result = None
                            detect_worker.embed_score = None

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
                    embed_out.reset()
                    print("[embed] Re-calibrating — keep swap OFF for a few seconds…")
                if was_swapping:
                    swap_flag.set()

    finally:
        stop_flag.set()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

