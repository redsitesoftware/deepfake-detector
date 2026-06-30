#!/usr/bin/env python3
"""preprocess_dataset.py — Build HDF5 dataset from real/fake face crop directories.

Applies video-level 80/10/10 train/val/test split (seed=42) to prevent leakage,
then packs all crops into a single data/dataset.h5 file.

Standard usage (self-generated DLC data):
    python scripts/preprocess_dataset.py \
        --real-dir data/real \
        --fake-dir data/fake \
        --out data/dataset.h5

DF40 usage (after download_datasets.py --df40 --df40-real):
    python scripts/preprocess_dataset.py \
        --df40-dir data/df40 \
        --out data/dataset.h5 \
        --size 380

    # Face-swap methods only (recommended — excludes talking-head/GAN):
    python scripts/preprocess_dataset.py \
        --df40-dir data/df40 \
        --methods simswap,deepfacelab,blendface,e4s,faceswap,hififace,infoswap,fsgan,megafs,facedancer \
        --out data/dataset.h5 \
        --size 380

The HDF5 layout:
    /train/images   (N, H, W, 3)  uint8
    /train/labels   (N,)          uint8  — 0=real, 1=fake
    /train/paths    (N,)          str
    /val/...
    /test/...
    /meta           attrs: real_count, fake_count, splits, created_at
"""
from __future__ import annotations

import argparse
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# DF40 face-swap methods (excludes talking-head and full-face synthesis)
DF40_FACESWAP_METHODS = {
    "simswap", "deepfacelab", "blendface", "e4s", "faceswap",
    "hififace", "infoswap", "fsgan", "megafs", "facedancer",
}


def _session_key(path: Path, base_dir: Path) -> str:
    """First subdirectory under base_dir → video/session key for splitting."""
    rel = path.relative_to(base_dir)
    return rel.parts[0] if len(rel.parts) > 1 else "default"


def _split_sessions(sessions: list[str], train=0.8, val=0.1, seed=42) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    n       = len(shuffled)
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


def _load_image(path: Path, size: int) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def _unzip_if_needed(zip_path: Path, dest_dir: Path) -> Path:
    """Unzip zip_path into dest_dir/<zip_stem>/ if not already done."""
    out = dest_dir / zip_path.stem
    sentinel = out / ".unzipped"
    if sentinel.exists():
        print(f"  Already unzipped: {out}")
        return out
    print(f"  Unzipping {zip_path.name} → {out} …")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)
    sentinel.touch()
    return out


def _collect_df40(df40_dir: Path, methods: set[str] | None
                  ) -> tuple[list[Path], list[Path]]:
    """Collect real and fake paths from a DF40 download directory.

    DF40 structure after gdown download:
        df40_dir/
          real/
            ff_real.zip           ← unzip to real/ff_real/
            celeb_real.zip        ← unzip to real/celeb_real/
          fake/
            blendface.zip         ← unzip to fake/blendface/
            simswap.zip           ← unzip to fake/simswap/
            ...
    """
    real_dir = df40_dir / "real"
    fake_dir = df40_dir / "fake"

    # Unzip real face crops
    real_paths: list[Path] = []
    for zp in sorted(real_dir.glob("*.zip")):
        extracted = _unzip_if_needed(zp, real_dir)
        real_paths += sorted(p for p in extracted.rglob("*")
                             if p.suffix.lower() in _IMG_EXTS)

    # Also pick up already-extracted images directly under real/
    for p in real_dir.rglob("*"):
        if p.suffix.lower() in _IMG_EXTS and ".unzipped" not in str(p):
            if p not in real_paths:
                real_paths.append(p)

    # Unzip fake method zips, optionally filtered by method name
    fake_paths: list[Path] = []
    if fake_dir.exists():
        for entry in sorted(fake_dir.iterdir()):
            # Determine method name whether it's a zip or already-extracted dir
            if entry.suffix.lower() == ".zip":
                name = entry.stem.lower()
            elif entry.is_dir():
                name = entry.name.lower()
            else:
                continue

            if methods and name not in methods:
                continue

            # Unzip if needed
            if entry.suffix.lower() == ".zip":
                entry = _unzip_if_needed(entry, fake_dir)
            if not entry.is_dir():
                continue

            method_paths = sorted(p for p in entry.rglob("*")
                                  if p.suffix.lower() in _IMG_EXTS)
            print(f"  fake/{name}: {len(method_paths)} images")
            fake_paths += method_paths

    return sorted(set(real_paths)), sorted(set(fake_paths))


def _write_hdf5(out_path: Path, real_paths: list[Path], fake_paths: list[Path],
                real_base: Path, fake_base: Path,
                size: int, train: float, val: float, seed: int) -> int:
    real_sessions = list({_session_key(p, real_base) for p in real_paths})
    fake_sessions = list({_session_key(p, fake_base) for p in fake_paths})

    real_split = _split_sessions(real_sessions, train, val, seed)
    fake_split = _split_sessions(fake_sessions, train, val, seed)

    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for p in real_paths:
        s = real_split.get(_session_key(p, real_base), "train")
        splits[s].append((p, 0))
    for p in fake_paths:
        s = fake_split.get(_session_key(p, fake_base), "train")
        splits[s].append((p, 1))

    for split_name, items in splits.items():
        r = sum(1 for _, l in items if l == 0)
        f = sum(1 for _, l in items if l == 1)
        print(f"  {split_name:5s}: {len(items):6d} total  (real={r}, fake={f})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0

    with h5py.File(str(out_path), "w") as f:
        for split_name, items in splits.items():
            random.Random(seed).shuffle(items)
            n = len(items)
            if n == 0:
                continue

            grp    = f.create_group(split_name)
            images = grp.create_dataset(
                "images", shape=(n, size, size, 3),
                dtype=np.uint8, compression="gzip", compression_opts=4)
            labels   = grp.create_dataset("labels",  shape=(n,), dtype=np.uint8)
            paths_ds = grp.create_dataset("paths",   shape=(n,),
                                          dtype=h5py.special_dtype(vlen=str))

            ok = 0
            for i, (path, label) in enumerate(items):
                img = _load_image(path, size)
                if img is None:
                    print(f"  WARN: skipping {path}")
                    continue
                images[ok] = img
                labels[ok] = label
                paths_ds[ok] = str(path)
                ok += 1
                if (i + 1) % 1000 == 0:
                    print(f"  [{split_name}] {i+1}/{n}…")

            if ok < n:
                grp["images"].resize((ok, size, size, 3))
                grp["labels"].resize((ok,))
                grp["paths"].resize((ok,))

            print(f"  {split_name}: {ok} frames written")
            total_written += ok

        meta = f.create_group("meta")
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["image_size"] = size
        meta.attrs["seed"]       = seed
        meta.attrs["real_total"] = len(real_paths)
        meta.attrs["fake_total"] = len(fake_paths)
        meta.attrs["train_frac"] = train
        meta.attrs["val_frac"]   = val

    return total_written


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Pack face crops into HDF5 for training.",
    )

    # Standard mode
    parser.add_argument("--real-dir", default=None,
                        help="Directory of real face crops (session subdirs)")
    parser.add_argument("--fake-dir", default=None,
                        help="Directory of fake face crops (source subdirs)")

    # DF40 mode
    parser.add_argument("--df40-dir", default=None,
                        help="Root of DF40 download (contains real/ and fake/). "
                             "Overrides --real-dir / --fake-dir.")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated DF40 method names to include "
                             "(default: all face-swap methods). "
                             f"Face-swap set: {','.join(sorted(DF40_FACESWAP_METHODS))}")
    parser.add_argument("--faceswap-only", action="store_true",
                        help="Shortcut: use only DF40 face-swap methods "
                             "(equivalent to --methods simswap,deepfacelab,...)")

    parser.add_argument("--out",   default="data/dataset.h5")
    parser.add_argument("--size",  type=int, default=224,
                        help="Output image size. Use 380 for EfficientNet-B4 native res.")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--val",   type=float, default=0.10)
    args = parser.parse_args()

    # ── DF40 mode ─────────────────────────────────────────────────────────────
    if args.df40_dir:
        df40_dir = Path(args.df40_dir)
        if not df40_dir.exists():
            print(f"ERROR: --df40-dir not found: {df40_dir}")
            return 1

        if args.faceswap_only:
            methods = DF40_FACESWAP_METHODS
        elif args.methods:
            methods = {m.strip().lower() for m in args.methods.split(",")}
        else:
            methods = DF40_FACESWAP_METHODS  # default: face-swap only for relevance

        print(f"[preprocess] DF40 mode — methods: {sorted(methods)}")
        real_paths, fake_paths = _collect_df40(df40_dir, methods)
        real_base = df40_dir / "real"
        fake_base = df40_dir / "fake"

    # ── Standard mode ─────────────────────────────────────────────────────────
    else:
        real_dir = Path(args.real_dir or "data/real")
        fake_dir = Path(args.fake_dir or "data/fake")
        real_paths = sorted(p for p in real_dir.rglob("*")
                            if p.suffix.lower() in _IMG_EXTS)
        fake_paths = sorted(p for p in fake_dir.rglob("*")
                            if p.suffix.lower() in _IMG_EXTS)
        real_base = real_dir
        fake_base = fake_dir

    if not real_paths:
        print("ERROR: no real images found"); return 1
    if not fake_paths:
        print("ERROR: no fake images found"); return 1

    print(f"[preprocess] Real: {len(real_paths)}  Fake: {len(fake_paths)}")
    if args.size == 224:
        print("  TIP: pass --size 380 for EfficientNet-B4 native resolution")

    out_path = Path(args.out)
    n = _write_hdf5(out_path, real_paths, fake_paths,
                    real_base, fake_base,
                    args.size, args.train, args.val, args.seed)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n✓ Dataset → {out_path}  ({size_mb:.0f} MB, {n} total frames)")
    print(f"\nNext steps:")
    print(f"  # Fine-tune from pre-trained FF++ model (recommended):")
    print(f"  python scripts/finetune_pretrained.py \\")
    print(f"      --checkpoint checkpoints/effnb4_ff_pretrained.pth \\")
    print(f"      --dataset {out_path} --epochs 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
