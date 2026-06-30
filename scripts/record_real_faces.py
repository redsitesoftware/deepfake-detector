#!/usr/bin/env python3
"""record_real_faces.py — Capture real face training data from webcam.

Records video and saves 1 face crop per second to data/real/.
Run multiple sessions (different lighting, angles, expressions) for diversity.

Usage:
    python scripts/record_real_faces.py --session dan_1 --duration 300
    python scripts/record_real_faces.py --session dan_2 --duration 300 --camera 1

Controls (while recording):
    Q / ESC  stop early
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session",  required=True,
                        help="Session name, e.g. 'dan_morning' — used in filenames")
    parser.add_argument("--duration", type=int, default=300,
                        help="Max recording seconds (default 300 = 5 min)")
    parser.add_argument("--camera",   type=int, default=0)
    parser.add_argument("--fps",      type=int, default=1,
                        help="Saved frames per second (default 1 — avoids near-duplicate frames)")
    parser.add_argument("--out-dir",  default="data/real")
    parser.add_argument("--size",     type=int, default=224,
                        help="Face crop output size (default 224 for EfficientNet)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.session
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy MTCNN import (facenet-pytorch)
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=args.size, margin=60, keep_all=False,
                  post_process=False, device="cpu")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        return 1

    saved      = 0
    skipped    = 0
    t_start    = time.time()
    t_last_save = -1.0
    interval   = 1.0 / args.fps

    print(f"\n── Recording session '{args.session}' ──────────────────────────")
    print(f"  Duration : {args.duration}s  |  Rate: {args.fps} fps  |  Out: {out_dir}")
    print("  Move naturally — vary angles, expressions, distance")
    print("  Q / ESC  stop early")
    print("────────────────────────────────────────────────────────────────\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - t_start
        if elapsed >= args.duration:
            break

        # Live preview with counter
        display = frame.copy()
        cv2.putText(display,
                    f"Recording… {int(elapsed)}s / {args.duration}s  |  saved: {saved}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 80), 2)
        cv2.imshow(f"Record real faces — {args.session}", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("\n[recorder] Stopped early by user")
            break

        # Save at the desired fps
        now = time.time()
        if now - t_last_save < interval:
            continue
        t_last_save = now

        # MTCNN face crop
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            crop_tensor = mtcnn(rgb)   # (3, 224, 224) float [0,255] or None
        except Exception:
            crop_tensor = None

        if crop_tensor is None:
            skipped += 1
            continue

        # Convert to uint8 BGR for saving
        crop_np = crop_tensor.permute(1, 2, 0).numpy()
        crop_np = np.clip(crop_np, 0, 255).astype(np.uint8)
        crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)

        fname = out_dir / f"{args.session}_{saved:05d}.jpg"
        cv2.imwrite(str(fname), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    cv2.destroyAllWindows()

    total = time.time() - t_start
    print(f"\n[recorder] Done — {saved} faces saved, {skipped} skipped (no face detected)")
    print(f"  Session  : {args.session}")
    print(f"  Duration : {total:.0f}s")
    print(f"  Output   : {out_dir}")
    print(f"\nNext: run more sessions for diversity, then:")
    print(f"  python scripts/generate_fakes.py --real-dir data/real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
