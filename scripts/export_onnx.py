#!/usr/bin/env python3
"""Export a trained PyTorch checkpoint to ONNX (fp32 + fp16).

Supports two checkpoint formats:
  --model-type timm       train.py output (timm EfficientNet-B4, 1-class BCE, 224px)
  --model-type dfbench    finetune_pretrained.py output (efficientnet_pytorch, 2-class, 380px)

Usage:
    # After finetune_pretrained.py
    python scripts/export_onnx.py \\
        --checkpoint checkpoints/effnb4_finetuned_best.pt \\
        --model-type dfbench \\
        --version v1.0

    # After train.py (from scratch)
    python scripts/export_onnx.py \\
        --checkpoint checkpoints/best.pt \\
        --model-type timm \\
        --version v1.0

The script writes:
    models/efficientnet_b4_v1_0.onnx
    models/efficientnet_b4_v1_0_fp16.onnx  (if onnxconverter-common available)

and prints SHA256 checksums to paste into models/manifest.json.
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


def _verify_onnx(path: Path, input_shape: tuple[int, ...]) -> bool:
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        dummy = np.random.rand(*input_shape).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: dummy})
        print(f"  ✓ onnxruntime inference OK  input={input_shape}  output={out[0].shape}")
        return True
    except ImportError:
        print("  ⚠ onnxruntime not installed — skipping verification")
        return True
    except Exception as e:
        print(f"  ✗ onnxruntime verification FAILED: {e}")
        return False


def _export_fp32(model, output_path: Path, input_shape: tuple[int, ...]) -> None:
    import torch
    model.eval()
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print(f"  Exported fp32 → {output_path}  ({output_path.stat().st_size // 1024 // 1024} MB)")


def _export_fp16(fp32_path: Path, fp16_path: Path) -> bool:
    try:
        import onnx
        from onnxconverter_common import float16
        model = onnx.load(str(fp32_path))
        fp16_model = float16.convert_float_to_float16(model)
        onnx.save(fp16_model, str(fp16_path))
        print(f"  Exported fp16 → {fp16_path}  ({fp16_path.stat().st_size // 1024 // 1024} MB)")
        return True
    except ImportError:
        print("  ⚠ onnxconverter-common not installed — skipping fp16. "
              "Install: pip install onnxconverter-common onnx")
        return False


def _load_timm_model(checkpoint_path: Path):
    """Load a timm EfficientNet-B4 model from train.py checkpoint."""
    import timm, torch
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)  # train.py saves {"model": state_dict, ...}
    if not isinstance(state, dict):
        raise ValueError("Expected a state-dict under 'model' key or bare state-dict")
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1)
    model.load_state_dict(state)
    print(f"  Loaded timm efficientnet_b4  (1-class, sigmoid output)")
    return model, (1, 3, 224, 224)


def _load_dfbench_model(checkpoint_path: Path):
    """Load a DeepfakeBench-format EfficientNet-B4 from finetune_pretrained.py checkpoint."""
    import sys, torch
    sys.path.insert(0, str(Path(__file__).parent))
    from finetune_pretrained import DeepfakeDetectorB4, load_finetuned_checkpoint

    device = torch.device("cpu")
    ckpt   = torch.load(checkpoint_path, map_location="cpu")

    model  = DeepfakeDetectorB4(dropout=0.0)  # dropout=0 for export (deterministic)

    if "model" in ckpt:
        # Fine-tuned checkpoint from finetune_pretrained.py
        model.load_state_dict(ckpt["model"])
        val_auc = ckpt.get("val_auc", 0)
        print(f"  Loaded fine-tuned dfbench model  (val_auc={val_auc:.4f})")
    else:
        # Original DeepfakeBench effnb4_best.pth
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded original DeepfakeBench checkpoint")

    print(f"  Architecture: EfficientNet-B4, 2-class softmax output [real, fake]")
    return model, (1, 3, 380, 380)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export EfficientNet-B4 checkpoint to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", choices=["timm", "dfbench"], default="dfbench",
                        help="Checkpoint format: 'dfbench' (finetune_pretrained.py) "
                             "or 'timm' (train.py). Default: dfbench")
    parser.add_argument("--version",    default="v1.0")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--model-name", default="efficientnet_b4_deepfake")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed", file=sys.stderr)
        return 1

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ver_tag  = args.version.replace(".", "_")
    fp32_path = out_dir / f"efficientnet_b4_{ver_tag}.onnx"
    fp16_path = out_dir / f"efficientnet_b4_{ver_tag}_fp16.onnx"

    print(f"\n=== Loading {args.model_type} checkpoint: {ckpt_path} ===")
    if args.model_type == "dfbench":
        model, input_shape = _load_dfbench_model(ckpt_path)
    else:
        model, input_shape = _load_timm_model(ckpt_path)

    print(f"\n=== Exporting fp32 (input: {input_shape}) ===")
    _export_fp32(model, fp32_path, input_shape)

    print("\n=== Verifying fp32 ===")
    if not _verify_onnx(fp32_path, input_shape):
        return 1

    print(f"\n=== Exporting fp16 ===")
    has_fp16 = _export_fp16(fp32_path, fp16_path)
    if has_fp16:
        print("\n=== Verifying fp16 ===")
        _verify_onnx(fp16_path, input_shape)

    print("\n=== SHA256 (paste into models/manifest.json) ===")
    print(f'  "{fp32_path.name}": "{_sha256(fp32_path)}"')
    if has_fp16:
        print(f'  "{fp16_path.name}": "{_sha256(fp16_path)}"')

    print(f"\n✓ Done. Update models/manifest.json then test with:")
    print(f"  python -c \"from deepfake_detector.cnn.detector import CNNDetector; "
          f"d=CNNDetector(); d.warmup(); print('CNN ready')\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
