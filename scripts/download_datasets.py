#!/usr/bin/env python3
"""download_datasets.py — Download deepfake training datasets without approval forms.

Datasets available right now (no form required):
  --ff-real      FaceForensics++ real videos via yt-dlp (conversion_dict.json public)
  --df40         DF40 face crops from Google Drive (40 methods, no approval)
  --df40-real    DF40 real face crops (FF++ + Celeb-DF real domain)
  --dfbench      DeepfakeBench pre-trained EfficientNet-B4 checkpoint (GitHub Releases)

Datasets requiring form approval:
  FF++  fakes:  https://bit.ly/faceforensics-form   (~2-7 business days)
  Celeb-DF v2:  https://forms.gle/2jYBby6y1FBU3u6q9 (~1-5 days)
  DFDC:         https://ai.meta.com/datasets/dfdc/  (requires AWS IAM setup)

Usage:
    python scripts/download_datasets.py --dfbench
    python scripts/download_datasets.py --df40 --df40-real
    python scripts/download_datasets.py --ff-real --output data/ff_real --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

FF_DICT_URL = (
    "https://raw.githubusercontent.com/ondyari/FaceForensics/"
    "master/dataset/conversion_dict.json"
)

DFBENCH_EFFNB4_URL = (
    "https://github.com/SCLBD/DeepfakeBench/releases/download/"
    "v1.0.1/effnb4_best.pth"
)

# DF40 Google Drive — no approval, direct public share
# Training fake face crops (~50 GB total across all 40 methods)
DF40_TRAIN_FOLDER_ID = "1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD"
# Testing fake face crops (~93 GB)
DF40_TEST_FOLDER_ID  = "1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_"
# Real domain: FF++ source crops
DF40_REAL_FFPP_ID    = "1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb"
# Real domain: Celeb-DF crops
DF40_REAL_CELEB_ID   = "1FGZ3aYsF-Yru50rPLoT5ef8-2Nkt4uBw"
# DF40 pre-trained weights (10 models, ~500 MB)
DF40_WEIGHTS_FOLDER  = "1HDgIOutGw3jsFXwvSQYeDoVPAzgfYbyr"

# Face-swap method IDs within DF40 training folder (skip talking-head methods)
DF40_FACESWAP_METHODS = [
    "simswap", "deepfacelab", "blendface", "e4s",
    "faceswap", "hififace", "infoswap", "fsgan",
    "megafs", "facedancer",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _check_tool(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True).returncode == 0


def _download_url(url: str, dest: Path, label: str = "") -> bool:
    label = label or dest.name
    print(f"[download] {label} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done  = 0
            chunk = 1 << 16
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                done += len(data)
                if total:
                    pct = done / total * 100
                    print(f"\r  {pct:.0f}%  ({done//1024//1024} MB)", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def _gdown(file_id: str, dest: Path, label: str = "") -> bool:
    """Download a single file from Google Drive using gdown or requests."""
    label = label or dest.name
    print(f"[gdown] {label} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _check_tool("gdown"):
        r = subprocess.run(
            ["gdown", "--id", file_id, "-O", str(dest)],
            capture_output=False,
        )
        return r.returncode == 0

    # Fallback: direct download URL (works for smaller files < ~100 MB)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return _download_url(url, dest, label)


def _gdown_folder(folder_id: str, dest_dir: Path, label: str = "") -> bool:
    """Download entire Google Drive folder using gdown."""
    label = label or dest_dir.name
    print(f"[gdown-folder] {label} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not _check_tool("gdown"):
        print("  ERROR: gdown not installed. Run: pip install gdown")
        return False

    r = subprocess.run(
        ["gdown", "--folder", f"https://drive.google.com/drive/folders/{folder_id}",
         "-O", str(dest_dir), "--remaining-ok"],
        capture_output=False,
    )
    return r.returncode == 0


# ── Dataset downloaders ───────────────────────────────────────────────────────

def download_dfbench_weights(output_dir: Path) -> bool:
    """Download DeepfakeBench pre-trained EfficientNet-B4 weights from GitHub Releases."""
    dest = output_dir / "effnb4_ff_pretrained.pth"
    if dest.exists():
        print(f"[dfbench] Already exists: {dest}")
        return True
    return _download_url(DFBENCH_EFFNB4_URL, dest, "EfficientNet-B4 (FF++ pretrained, 67 MB)")


def download_ff_real(output_dir: Path, limit: int | None = None) -> int:
    """Download FaceForensics++ real videos via yt-dlp.

    The conversion_dict.json file is publicly readable from GitHub.
    It maps sequence IDs (000-977) to YouTube video IDs — no form required.

    Returns number of successfully downloaded videos.
    """
    if not _check_tool("yt-dlp"):
        print("ERROR: yt-dlp not found. Install: pip install yt-dlp")
        return 0

    # 1. Download the public conversion dict
    dict_path = output_dir / "conversion_dict.json"
    if not dict_path.exists():
        ok = _download_url(FF_DICT_URL, dict_path, "FF++ conversion_dict.json")
        if not ok:
            return 0

    conv = json.loads(dict_path.read_text())
    print(f"[ff-real] {len(conv)} sequences in conversion dict")

    output_dir.mkdir(parents=True, exist_ok=True)
    items = list(conv.items())
    if limit:
        items = items[:limit]

    ok_count = 0
    fail_count = 0
    skip_count = 0

    for seq, val in items:
        ytid = val.split()[0]
        dest = output_dir / f"{seq}.mp4"

        if dest.exists() and dest.stat().st_size > 10_000:
            skip_count += 1
            continue

        url = f"https://www.youtube.com/watch?v={ytid}"
        r = subprocess.run(
            ["yt-dlp",
             "-f", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
             "--merge-output-format", "mp4",
             "-o", str(dest),
             "--quiet",
             "--no-warnings",
             url],
            capture_output=True,
            timeout=120,
        )
        if r.returncode == 0 and dest.exists():
            ok_count += 1
            print(f"  [{ok_count+skip_count}/{len(items)}] {seq} ✓")
        else:
            fail_count += 1
            err = r.stderr.decode(errors="ignore")[:80]
            print(f"  [{ok_count+skip_count}/{len(items)}] {seq} ✗  {err}")

    print(f"\n[ff-real] Done: {ok_count} downloaded, {skip_count} skipped, {fail_count} failed")
    print(f"  → Run generate_fakes.py on these to create face-swap training pairs")
    return ok_count


def download_df40(output_dir: Path, faceswap_only: bool = True,
                  include_real: bool = True) -> bool:
    """Download DF40 pre-processed face crops (no approval required).

    DF40 is a NeurIPS 2024 benchmark with 40 deepfake methods.
    The training data (~50 GB) is on Google Drive, publicly shared.
    
    Args:
        faceswap_only: Only download face-swap methods (skip talking-head/GAN synthesis)
        include_real:  Also download the real face crops (FF++ + Celeb-DF domains)
    """
    if not _check_tool("gdown"):
        print("Installing gdown…")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"])

    ok = True

    if include_real:
        print("\n[df40] Downloading real face crops…")
        ok &= _gdown(DF40_REAL_FFPP_ID,  output_dir / "real" / "ff_real.zip",
                     "DF40 real faces (FF++ domain)")
        ok &= _gdown(DF40_REAL_CELEB_ID, output_dir / "real" / "celeb_real.zip",
                     "DF40 real faces (Celeb-DF domain)")

    print("\n[df40] Downloading fake face crops…")
    if faceswap_only:
        print(f"  Face-swap methods only: {DF40_FACESWAP_METHODS}")
        # Attempt to download the full folder — gdown will get all subfolders
        # including face-swap methods. Filter by method name in preprocess_dataset.py
        ok &= _gdown_folder(DF40_TRAIN_FOLDER_ID, output_dir / "fake",
                            "DF40 training fakes (all 40 methods, ~50 GB)")
        print("\n  NOTE: DF40 folder contains all 40 methods.")
        print(f"  Use --methods flag in preprocess_dataset.py to filter face-swap only:")
        print(f"  --methods {','.join(DF40_FACESWAP_METHODS)}")
    else:
        ok &= _gdown_folder(DF40_TRAIN_FOLDER_ID, output_dir / "fake",
                            "DF40 training fakes (~50 GB)")

    if ok:
        print(f"\n[df40] ✓ Downloaded to {output_dir}")
        print("  Next: python scripts/preprocess_dataset.py --input-dir", output_dir,
              "--output data/dataset.h5")
    return ok


def download_df40_weights(output_dir: Path) -> bool:
    """Download DF40 pre-trained detector weights (10 models, Google Drive)."""
    return _gdown_folder(DF40_WEIGHTS_FOLDER, output_dir / "df40_pretrained",
                         "DF40 pre-trained weights (~500 MB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download deepfake training datasets (no-approval options only by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", default="data", help="Output root directory")
    parser.add_argument("--dfbench", action="store_true",
                        help="Download DeepfakeBench EfficientNet-B4 pre-trained weights")
    parser.add_argument("--ff-real", action="store_true",
                        help="Download FF++ real videos via yt-dlp (~750 videos, ~25 GB)")
    parser.add_argument("--ff-limit", type=int, default=None, metavar="N",
                        help="Limit number of FF++ videos (default: all ~977)")
    parser.add_argument("--df40", action="store_true",
                        help="Download DF40 fake face crops (~50 GB, Google Drive, no form)")
    parser.add_argument("--df40-real", action="store_true",
                        help="Download DF40 real face crops (FF++ + Celeb-DF domains)")
    parser.add_argument("--df40-weights", action="store_true",
                        help="Download DF40 pre-trained detector weights")
    parser.add_argument("--all-no-approval", action="store_true",
                        help="Download all datasets that don't require a form")
    args = parser.parse_args()

    if args.all_no_approval:
        args.dfbench = True
        args.ff_real = True
        args.df40    = True
        args.df40_real = True

    if not any([args.dfbench, args.ff_real, args.df40,
                args.df40_real, args.df40_weights]):
        parser.print_help()
        print("\nTip: start with --dfbench to get the pre-trained FF++ model (67 MB, instant)")
        return 1

    output = Path(args.output)
    any_ok = False

    if args.dfbench:
        ok = download_dfbench_weights(output / "checkpoints")
        if ok:
            print(f"\n✓ Pre-trained model at: {output}/checkpoints/effnb4_ff_pretrained.pth")
            print("  Test it: python scripts/finetune_pretrained.py --eval-only --checkpoint",
                  f"{output}/checkpoints/effnb4_ff_pretrained.pth")
        any_ok |= ok

    if args.ff_real:
        n = download_ff_real(output / "ff_real", limit=args.ff_limit)
        any_ok |= (n > 0)

    if args.df40 or args.df40_real:
        ok = download_df40(output / "df40",
                           faceswap_only=True,
                           include_real=args.df40_real)
        any_ok |= ok

    if args.df40_weights:
        ok = download_df40_weights(output)
        any_ok |= ok

    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
