#!/usr/bin/env python3
"""generate_fakes.py — Batch-generate fake face crops using Deep-Live-Cam.

Reads real face images from data/real/, applies DLC face-swap with a set of
source faces, and writes MTCNN-cropped fake frames to data/fake/.

Usage:
    python scripts/generate_fakes.py \
        --real-dir data/real \
        --source-dir ~/Desktop/Deep\ Fake\ Tests \
        --dlc ~/PROJECTS/Deep-Live-Cam

Generates N_SOURCES × len(real_frames) fake crops.  Caps output per source
at --max-per-source (default 600) to balance the dataset.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_dlc(dlc_path: str):
    """Bootstrap DLC modules (same stub as dlc_bridge.py)."""
    sys.path.insert(0, dlc_path)
    stub = types.ModuleType("modules.ui")
    stub.update_status = lambda *a, **kw: None
    stub.check_and_ignore_nsfw = lambda *a, **kw: False
    sys.modules.setdefault("modules.ui", stub)

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
    g.fp_ui = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

    print("[gen] Loading face analyser…")
    from modules.face_analyser import get_face_analyser, get_one_face
    from modules import imread_unicode
    get_face_analyser()

    print("[gen] Loading face swapper model…")
    from modules.processors.frame.core import get_frame_processors_modules
    from modules.face_analyser import detect_one_face_fast
    fps_mods = get_frame_processors_modules(["face_swapper"])
    fp = next(fp for fp in fps_mods if fp.NAME == "DLC.FACE-SWAPPER")

    return fp, get_one_face, detect_one_face_fast, imread_unicode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-generate fake training frames with DLC face-swap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-dir",       default="data/real",
                        help="Directory containing real face crops (recursively scanned)")
    parser.add_argument("--source-dir",     required=True,
                        help="Directory of source (swap target) face images")
    parser.add_argument("--out-dir",        default="data/fake")
    parser.add_argument("--dlc",            default=str(Path.home() / "PROJECTS" / "Deep-Live-Cam"))
    parser.add_argument("--max-per-source", type=int, default=600,
                        help="Cap fake frames generated per source face")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect real frames
    real_dir = Path(args.real_dir)
    real_paths = sorted(p for p in real_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    if not real_paths:
        print(f"ERROR: no images found in {real_dir}")
        return 1
    print(f"[gen] Found {len(real_paths)} real frames")

    # Collect source faces
    src_dir = Path(args.source_dir)
    src_paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not src_paths:
        print(f"ERROR: no source images in {src_dir}")
        return 1
    print(f"[gen] Found {len(src_paths)} source faces")

    # Load DLC
    if not Path(args.dlc).is_dir():
        print(f"ERROR: DLC not found at {args.dlc}")
        return 1

    fp, get_one_face, detect_one_face_fast, imread_unicode = _load_dlc(args.dlc)

    # MTCNN for output cropping
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=224, margin=60, keep_all=False,
                  post_process=False, device="cpu")

    out_dir = Path(args.out_dir)
    total_saved = 0
    total_skipped = 0

    for src_path in src_paths:
        src_img = imread_unicode(str(src_path))
        src_face = get_one_face(src_img)
        if src_face is None:
            print(f"[gen] No face in source {src_path.name} — skipping")
            continue

        src_name = src_path.stem
        src_out  = out_dir / src_name
        src_out.mkdir(parents=True, exist_ok=True)

        # Sample frames for this source (cap to max_per_source)
        frames_for_src = real_paths.copy()
        random.shuffle(frames_for_src)
        frames_for_src = frames_for_src[:args.max_per_source]

        saved   = 0
        skipped = 0
        t0      = time.time()

        print(f"\n[gen] Source: {src_path.name}  ({len(frames_for_src)} frames)")

        for i, real_path in enumerate(frames_for_src):
            # Load real frame (BGR)
            real_frame = cv2.imread(str(real_path))
            if real_frame is None:
                skipped += 1
                continue

            # Detect face in real frame
            target_face = detect_one_face_fast(real_frame)
            if target_face is None:
                skipped += 1
                continue

            # Apply swap
            try:
                swapped = fp.swap_face(src_face, target_face, real_frame.copy())
                bboxes  = [target_face.bbox.astype(int)] if hasattr(target_face, "bbox") and target_face.bbox is not None else []
                swapped = fp.apply_post_processing(swapped, bboxes)
            except Exception as e:
                skipped += 1
                continue

            # MTCNN crop the swapped output
            rgb = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)
            try:
                crop_tensor = mtcnn(rgb)
            except Exception:
                crop_tensor = None

            if crop_tensor is None:
                skipped += 1
                continue

            crop_np  = crop_tensor.permute(1, 2, 0).numpy()
            crop_np  = np.clip(crop_np, 0, 255).astype(np.uint8)
            crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)

            fname = src_out / f"{src_name}_{saved:05d}.jpg"
            cv2.imwrite(str(fname), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                fps_rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(frames_for_src)}]  saved={saved}  "
                      f"skip={skipped}  {fps_rate:.1f} fr/s")

        total_saved   += saved
        total_skipped += skipped
        print(f"  Done — {saved} saved, {skipped} skipped")

    print(f"\n[gen] Complete: {total_saved} total fake frames in {out_dir}/")
    print(f"[gen] Skipped: {total_skipped} frames (no face detected / swap error)")
    print(f"\nNext: python scripts/preprocess_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
