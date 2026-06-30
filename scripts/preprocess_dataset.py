#!/usr/bin/env python3
"""preprocess_dataset.py — Build HDF5 dataset from real/fake face crop directories.

Applies video-level 80/10/10 train/val/test split (seed=42) to prevent leakage,
then packs all crops into a single data/dataset.h5 file.

Usage:
    python scripts/preprocess_dataset.py \
        --real-dir data/real \
        --fake-dir data/fake \
        --out data/dataset.h5

The HDF5 layout:
    /train/images   (N, 224, 224, 3)  uint8
    /train/labels   (N,)              uint8  — 0=real, 1=fake
    /train/paths    (N,)              str
    /val/...
    /test/...
    /meta           attrs: real_count, fake_count, splits, created_at
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _session_key(path: Path, base_dir: Path) -> str:
    """Return a session/video-level key for splitting (parent dir name = session)."""
    rel = path.relative_to(base_dir)
    return rel.parts[0] if len(rel.parts) > 1 else "default"


def _split_sessions(sessions: list[str], train=0.8, val=0.1, seed=42) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train))
    n_val   = max(1, int(n * val))
    mapping: dict[str, str] = {}
    for s in shuffled[:n_train]:
        mapping[s] = "train"
    for s in shuffled[n_train:n_train + n_val]:
        mapping[s] = "val"
    for s in shuffled[n_train + n_val:]:
        mapping[s] = "test"
    return mapping


def _load_image(path: Path, size: int = 224) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real-dir", default="data/real")
    parser.add_argument("--fake-dir", default="data/fake")
    parser.add_argument("--out",      default="data/dataset.h5")
    parser.add_argument("--size",     type=int, default=224)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--train",    type=float, default=0.80)
    parser.add_argument("--val",      type=float, default=0.10)
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)

    real_paths = sorted(p for p in real_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    fake_paths = sorted(p for p in fake_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)

    if not real_paths:
        print(f"ERROR: no real images in {real_dir}"); return 1
    if not fake_paths:
        print(f"ERROR: no fake images in {fake_dir}"); return 1

    print(f"Real frames : {len(real_paths)}")
    print(f"Fake frames : {len(fake_paths)}")

    # Video-level split — group by session (subdirectory name)
    real_sessions = list({_session_key(p, real_dir) for p in real_paths})
    fake_sessions = list({_session_key(p, fake_dir) for p in fake_paths})

    real_split = _split_sessions(real_sessions, args.train, args.val, args.seed)
    fake_split = _split_sessions(fake_sessions, args.train, args.val, args.seed)

    # Collect per-split
    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for p in real_paths:
        s = real_split.get(_session_key(p, real_dir), "train")
        splits[s].append((p, 0))
    for p in fake_paths:
        s = fake_split.get(_session_key(p, fake_dir), "train")
        splits[s].append((p, 1))

    for split_name, items in splits.items():
        real_n = sum(1 for _, l in items if l == 0)
        fake_n = sum(1 for _, l in items if l == 1)
        print(f"  {split_name:5s}: {len(items):5d} total  (real={real_n}, fake={fake_n})")

    # Write HDF5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_path), "w") as f:
        for split_name, items in splits.items():
            random.Random(args.seed).shuffle(items)
            n = len(items)
            if n == 0:
                continue

            grp    = f.create_group(split_name)
            images = grp.create_dataset("images", shape=(n, args.size, args.size, 3),
                                        dtype=np.uint8, compression="gzip", compression_opts=4)
            labels = grp.create_dataset("labels", shape=(n,), dtype=np.uint8)
            paths_ds = grp.create_dataset("paths",  shape=(n,), dtype=h5py.special_dtype(vlen=str))

            ok = 0
            for i, (path, label) in enumerate(items):
                img = _load_image(path, args.size)
                if img is None:
                    print(f"  WARN: could not load {path}")
                    continue
                images[ok] = img
                labels[ok] = label
                paths_ds[ok] = str(path)
                ok += 1
                if (i + 1) % 500 == 0:
                    print(f"  [{split_name}] {i+1}/{n}…")

            # Trim if any failed loads
            if ok < n:
                grp["images"].resize((ok, args.size, args.size, 3))
                grp["labels"].resize((ok,))

            print(f"  {split_name}: {ok} frames written")

        # Metadata
        meta = f.create_group("meta")
        meta.attrs["created_at"]   = datetime.now(timezone.utc).isoformat()
        meta.attrs["image_size"]   = args.size
        meta.attrs["seed"]         = args.seed
        meta.attrs["real_total"]   = len(real_paths)
        meta.attrs["fake_total"]   = len(fake_paths)
        meta.attrs["train_frac"]   = args.train
        meta.attrs["val_frac"]     = args.val

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n✓ Dataset written → {out_path}  ({size_mb:.0f} MB)")
    print(f"\nNext: python scripts/train.py --config configs/efficientnet_b4_mvp.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
