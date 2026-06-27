#!/usr/bin/env python3
"""Pre-download all (or selected) registered models to a local cache directory.

Intended for CI and Docker build-time pre-population so that inference pods
start instantly without hitting HuggingFace Hub at runtime.

Usage:
    # Download all models
    python scripts/download_models.py

    # Download specific models only
    python scripts/download_models.py --models efficientnet_b4_deepfake,face_detector_mtcnn

    # Custom cache directory
    python scripts/download_models.py --cache-dir /opt/models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepfake_detector.models import MODEL_REGISTRY, ChecksumMismatchError, download
from deepfake_detector.models.registry import ModelNotFoundError


def _human_bytes(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download deepfake-detector model weights from HuggingFace Hub."
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated list of model names to download.  "
             "Omit to download all registered models.",
    )
    parser.add_argument(
        "--version",
        default="v1.0",
        help="Version tag to download for each model (default: v1.0).",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Override cache directory (default: ~/.cache/deepfake_detector/models/).",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        requested = list(MODEL_REGISTRY.keys())

    if not requested:
        print("No models to download.", file=sys.stderr)
        sys.exit(1)

    print(f"Models to download: {', '.join(requested)}")
    print(f"Version: {args.version}")
    if cache_dir:
        print(f"Cache dir: {cache_dir}")
    print()

    total_bytes = 0
    failures: list[str] = []

    for model_name in requested:
        try:
            path = download(model_name, version=args.version, cache_dir=cache_dir)
            size_str = _human_bytes(path)
            total_bytes += path.stat().st_size
            print(f"  ✓ {model_name}  ({size_str})  → {path}")
        except ModelNotFoundError as exc:
            print(f"  ✗ {model_name}: {exc}", file=sys.stderr)
            failures.append(model_name)
        except ChecksumMismatchError as exc:
            print(f"  ✗ {model_name}: checksum error — {exc}", file=sys.stderr)
            failures.append(model_name)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {model_name}: unexpected error — {exc}", file=sys.stderr)
            failures.append(model_name)

    total_mb = total_bytes / (1024 * 1024)
    print(f"\nTotal downloaded: {total_mb:.1f} MB across {len(requested) - len(failures)} model(s)")

    if failures:
        print(f"\nFailed downloads ({len(failures)}): {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)

    print("All models downloaded successfully.")


if __name__ == "__main__":
    main()
