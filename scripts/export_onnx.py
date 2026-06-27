#!/usr/bin/env python3
"""Export a trained PyTorch checkpoint to ONNX (fp32 + fp16).

Usage:
    python scripts/export_onnx.py \\
        --checkpoint path/to/weights.pt \\
        --model-name efficientnet_b4_deepfake \\
        --version v1.0

The script writes:
    models/efficientnet_b4_v1.0.onnx       (fp32)
    models/efficientnet_b4_v1.0_fp16.onnx  (fp16)

It verifies each output with a dummy onnxruntime inference pass and prints the
SHA256 checksums so they can be pasted into models/manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_onnx(path: Path, input_shape: tuple[int, ...]) -> None:
    """Run a dummy inference to confirm the ONNX graph is valid."""
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        dummy = np.random.rand(*input_shape).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        sess.run(None, {input_name: dummy})
        print(f"  ✓ onnxruntime inference OK ({input_name} shape={input_shape})")
    except ImportError:
        print("  ⚠ onnxruntime not installed — skipping inference verification")


def _export_fp32(model: object, output_path: Path, input_shape: tuple[int, ...]) -> None:
    import torch

    dummy = torch.randn(*input_shape)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"  Exported fp32 → {output_path}")


def _export_fp16(fp32_path: Path, fp16_path: Path) -> None:
    try:
        from onnxmltools.utils import load_model, save_model
        from onnxmltools.utils.float16_converter import convert_float_to_float16

        model = load_model(str(fp32_path))
        fp16_model = convert_float_to_float16(model)
        save_model(fp16_model, str(fp16_path))
        print(f"  Exported fp16 → {fp16_path}")
    except ImportError:
        # Fallback: use onnxconverter-common if available
        try:
            import onnx
            from onnxconverter_common import float16

            model = onnx.load(str(fp32_path))
            fp16_model = float16.convert_float_to_float16(model)
            onnx.save(fp16_model, str(fp16_path))
            print(f"  Exported fp16 (via onnxconverter-common) → {fp16_path}")
        except ImportError:
            print(
                "  ⚠ Neither onnxmltools nor onnxconverter-common installed — "
                "skipping fp16 export.  Install one to enable fp16."
            )
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a PyTorch checkpoint to ONNX fp32 + fp16."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt/.pth weights file")
    parser.add_argument(
        "--model-name",
        default="efficientnet_b4_deepfake",
        help="Registry model name (default: efficientnet_b4_deepfake)",
    )
    parser.add_argument("--version", default="v1.0", help="Version tag (default: v1.0)")
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to write ONNX files (default: models/)",
    )
    parser.add_argument(
        "--input-shape",
        default="1,3,224,224",
        help="Comma-separated NCHW input shape (default: 1,3,224,224)",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required for ONNX export.  pip install torch", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    version_tag = args.version.replace(".", "_")
    base_name = f"{args.model_name.replace('_deepfake', '')}_{version_tag}"
    fp32_path = output_dir / f"{base_name}.onnx"
    fp16_path = output_dir / f"{base_name}_fp16.onnx"

    print(f"\n=== Loading checkpoint: {checkpoint_path} ===")
    state = torch.load(checkpoint_path, map_location="cpu")
    model = state.get("model") or state  # support both bare state-dicts and full checkpoints

    if isinstance(model, dict):
        print("ERROR: checkpoint is a state-dict without an embedded model object. "
              "Please pass a full checkpoint with the model instance.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Exporting fp32 → {fp32_path} ===")
    _export_fp32(model, fp32_path, input_shape)

    print("\n=== Verifying fp32 ===")
    _verify_onnx(fp32_path, input_shape)

    print(f"\n=== Exporting fp16 → {fp16_path} ===")
    _export_fp16(fp32_path, fp16_path)
    if fp16_path.exists():
        print("\n=== Verifying fp16 ===")
        _verify_onnx(fp16_path, input_shape)

    print("\n=== SHA256 checksums (paste into models/manifest.json) ===")
    print(f"  {fp32_path.name}: {_sha256(fp32_path)}")
    if fp16_path.exists():
        print(f"  {fp16_path.name}: {_sha256(fp16_path)}")


if __name__ == "__main__":
    main()
