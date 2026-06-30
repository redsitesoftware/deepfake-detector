#!/usr/bin/env python3
"""finetune_pretrained.py — Fine-tune or evaluate the DeepfakeBench EfficientNet-B4.

Loads the pre-trained effnb4_best.pth from DeepfakeBench (trained on FF++)
and fine-tunes it on your custom data (DLC-generated deepfakes + real faces).

The pre-trained model already detects classic deepfakes well.
Fine-tuning on DLC (inswapper_128.onnx / InsightFace) outputs improves accuracy
for modern face-swap tools that post-date the FF++ dataset.

Architecture:
    backbone.efficientnet  — EfficientNet-B4 (efficientnet_pytorch library)
    backbone.last_layer    — Linear(1792, 2)  [class 0 = real, class 1 = fake]

Usage:
    # Evaluate pre-trained model zero-shot on your HDF5 dataset
    python scripts/finetune_pretrained.py \\
        --checkpoint checkpoints/effnb4_ff_pretrained.pth \\
        --dataset    data/dataset.h5 \\
        --eval-only

    # Fine-tune on DLC-generated data (recommended: freeze backbone first 5 epochs)
    python scripts/finetune_pretrained.py \\
        --checkpoint checkpoints/effnb4_ff_pretrained.pth \\
        --dataset    data/dataset.h5 \\
        --epochs     20 \\
        --freeze-backbone-epochs 5

    # Evaluate a saved fine-tuned checkpoint
    python scripts/finetune_pretrained.py \\
        --checkpoint checkpoints/effnb4_finetuned_best.pt \\
        --dataset    data/dataset.h5 \\
        --eval-only --is-finetuned
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Model ─────────────────────────────────────────────────────────────────────

class _EfficientNetB4Backbone(nn.Module):
    """Exact architecture used by DeepfakeBench's effnb4_best.pth."""

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        # Replace the original ImageNet head with Identity (backbone.last_layer is the head)
        self.efficientnet._fc = nn.Identity()
        self.last_layer = nn.Linear(1792, 2)
        self._dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.efficientnet.extract_features(x)             # (B, 1792, h, w)
        f = self.efficientnet._avg_pooling(f).flatten(1)       # (B, 1792)
        f = nn.functional.dropout(f, p=self._dropout_p, training=self.training)
        return self.last_layer(f)                               # (B, 2)


class DeepfakeDetectorB4(nn.Module):
    """Wrapper matching DeepfakeBench checkpoint structure (backbone.*)."""

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.backbone = _EfficientNetB4Backbone(dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze EfficientNet body — only train the classification head."""
        for p in self.backbone.efficientnet.parameters():
            p.requires_grad_(False)
        print("[model] Backbone frozen — only last_layer will be trained")

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for p in self.backbone.efficientnet.parameters():
            p.requires_grad_(True)
        print("[model] Backbone unfrozen — full model fine-tuning")


def load_dfbench_checkpoint(path: str | Path, device: torch.device,
                             dropout: float = 0.4) -> DeepfakeDetectorB4:
    """Load the DeepfakeBench effnb4_best.pth checkpoint.
    
    The checkpoint keys match DeepfakeBench's EfficientNetB4 backbone:
        backbone.efficientnet.*  — encoder layers
        backbone.last_layer.*   — 2-class head (0=real, 1=fake)
    """
    model = DeepfakeDetectorB4(dropout=dropout)
    ckpt  = torch.load(path, map_location="cpu")

    # Checkpoint has no _fc (replaced by Identity + last_layer), so strict=False
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    expected_missing = {"backbone.efficientnet._fc.weight",
                        "backbone.efficientnet._fc.bias"}
    real_missing = set(missing) - expected_missing
    if real_missing:
        print(f"  WARN: Unexpected missing keys: {real_missing}")
    if unexpected:
        print(f"  WARN: Unexpected keys in checkpoint: {unexpected}")

    return model.to(device)


def load_finetuned_checkpoint(path: str | Path, device: torch.device,
                               dropout: float = 0.4) -> DeepfakeDetectorB4:
    """Load a fine-tuned checkpoint saved by this script."""
    model = DeepfakeDetectorB4(dropout=dropout)
    ckpt  = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded fine-tuned: epoch={ckpt.get('epoch','?')} "
          f"val_auc={ckpt.get('val_auc',0):.4f}")
    return model.to(device)


# ── Dataset ───────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_INPUT_SIZE    = 380   # EfficientNet-B4 native resolution


def _build_transform(augment: bool) -> A.Compose:
    ops: list = []
    if augment:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
            A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.2),
        ]
    ops += [
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(ops)


class FaceDataset(Dataset):
    def __init__(self, h5_path: str, split: str, transform=None):
        self._h5_path  = h5_path
        self._split    = split
        self._transform = transform
        self._h5: h5py.File | None = None
        with h5py.File(h5_path, "r") as f:
            self._len    = len(f[split]["labels"])
            self._labels = f[split]["labels"][:]

    def __len__(self):
        return self._len

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")

    def __getitem__(self, idx):
        self._open()
        img   = self._h5[self._split]["images"][idx]  # (H, W, 3) uint8 RGB
        label = int(self._labels[idx])
        if self._transform:
            img = self._transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, label


# ── Train / eval helpers ──────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device).long()   # CrossEntropyLoss expects long

            logits = model(imgs)                 # (B, 2)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(imgs)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(fake)
            all_probs.extend(probs.detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    mean_loss = total_loss / max(len(all_labels), 1)
    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.5

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc   = sum(p == l for p, l in zip(preds, all_labels)) / max(len(all_labels), 1)
    return mean_loss, auc, acc


def evaluate(model, loader, device) -> dict:
    """Run evaluation without a criterion — for zero-shot testing."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs  = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc   = sum(p == l for p, l in zip(preds, all_labels)) / max(len(all_labels), 1)
    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.5

    n_real  = sum(1 for l in all_labels if l == 0)
    n_fake  = sum(1 for l in all_labels if l == 1)
    tp = sum(1 for p, l in zip(preds, all_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, all_labels) if p == 0 and l == 1)

    return dict(auc=auc, acc=acc, tp=tp, tn=tn, fp=fp, fn=fn,
                n_real=n_real, n_fake=n_fake,
                tpr=tp / max(tp + fn, 1),
                tnr=tn / max(tn + fp, 1))


def _print_eval(m: dict, prefix: str = ""):
    print(f"{prefix}AUC={m['auc']:.4f}  ACC={m['acc']:.4f}  "
          f"TPR(fake recall)={m['tpr']:.4f}  TNR(real recall)={m['tnr']:.4f}")
    print(f"  Real: {m['n_real']}  Fake: {m['n_fake']}  "
          f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepfakeBench EfficientNet-B4 on custom DLC data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to effnb4_best.pth or fine-tuned .pt checkpoint")
    parser.add_argument("--dataset",    required=True,
                        help="HDF5 dataset from preprocess_dataset.py")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Only evaluate — do not fine-tune")
    parser.add_argument("--is-finetuned", action="store_true",
                        help="Checkpoint is a fine-tuned .pt (not original dfbench format)")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Learning rate (full backbone). Head-only uses lr*10")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5,
                        help="Freeze backbone for first N epochs (train head only)")
    parser.add_argument("--dropout",    type=float, default=0.4)
    parser.add_argument("--output-dir", default="checkpoints",
                        help="Where to save best fine-tuned checkpoint")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[finetune] Device: {device}")

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return 1

    print(f"[finetune] Loading checkpoint: {ckpt_path}")
    if args.is_finetuned:
        model = load_finetuned_checkpoint(ckpt_path, device, args.dropout)
    else:
        model = load_dfbench_checkpoint(ckpt_path, device, args.dropout)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Datasets
    val_tf   = _build_transform(augment=False)
    val_ds   = FaceDataset(args.dataset, "val",  val_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=False)
    print(f"[finetune] Val: {len(val_ds)} samples")

    # Zero-shot evaluation first
    print("\n[finetune] Zero-shot evaluation on val split:")
    m = evaluate(model, val_loader, device)
    _print_eval(m, "  ")

    if args.eval_only:
        return 0

    # Fine-tuning setup
    train_tf   = _build_transform(augment=True)
    train_ds   = FaceDataset(args.dataset, "train", train_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=False)
    print(f"[finetune] Train: {len(train_ds)} samples")

    # Class balance weight
    n_real = int((train_ds._labels == 0).sum())
    n_fake = int((train_ds._labels == 1).sum())
    # CrossEntropyLoss with weight=[1/n_real, 1/n_fake] normalized
    total = n_real + n_fake
    w = torch.tensor([total / (2 * n_real), total / (2 * n_fake)],
                     dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    print(f"  Class weights: real={w[0]:.3f} fake={w[1]:.3f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_auc  = m["auc"]  # start from zero-shot performance
    best_path = out_dir / "effnb4_finetuned_best.pt"

    print(f"\n[finetune] Starting {args.epochs} epochs "
          f"(freeze backbone for first {args.freeze_backbone_epochs})…\n")

    for epoch in range(args.epochs):
        t0 = time.time()

        # Phase control: freeze backbone early, unfreeze for full fine-tuning
        if epoch == 0 and args.freeze_backbone_epochs > 0:
            model.freeze_backbone()
            # Head-only optimizer with higher LR
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 10, weight_decay=1e-4)

        if epoch == args.freeze_backbone_epochs:
            model.unfreeze_backbone()
            # Full model optimizer with lower LR
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=1e-4)
            n_remaining = args.epochs - epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_remaining * len(train_loader), eta_min=1e-6)

        tr_loss, tr_auc, tr_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)

        if epoch >= args.freeze_backbone_epochs:
            scheduler.step()

        va_metrics = evaluate(model, val_loader, device)
        va_auc = va_metrics["auc"]
        elapsed = time.time() - t0

        marker = " ← best" if va_auc > best_auc else ""
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{args.epochs}  "
              f"tr_loss={tr_loss:.4f} tr_auc={tr_auc:.4f}  "
              f"va_auc={va_auc:.4f} va_acc={va_metrics['acc']:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s{marker}")

        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "val_auc":  va_auc,
                "val_metrics": va_metrics,
                "optimizer": optimizer.state_dict(),
            }, best_path)
            print(f"  ✓ Saved best checkpoint → {best_path}")

    print(f"\n[finetune] Done. Best val AUC: {best_auc:.4f}")
    print(f"  Checkpoint: {best_path}")
    print(f"  Export ONNX: python scripts/export_onnx.py --checkpoint {best_path} --is-finetuned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
