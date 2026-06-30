#!/usr/bin/env python3
"""train.py — Fine-tune EfficientNet-B4 on real/fake face dataset.

Usage:
    python scripts/train.py --config configs/efficientnet_b4_mvp.yaml

Checkpoints best model by val AUC (not loss — avoids DeepfakeBench #186 pitfall).
Logs to W&B if WANDB_PROJECT is set, otherwise stdout only.

Hardware: Apple M-series MPS (~2-4 hrs / 10 epochs on ~3k frames)
          CUDA (Colab A100): ~30-45 min
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Dataset ───────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    def __init__(self, h5_path: str, split: str, transform=None):
        self._h5_path  = h5_path
        self._split    = split
        self._transform = transform
        self._h5: h5py.File | None = None

        # Read length without keeping file open (fork-safe)
        with h5py.File(h5_path, "r") as f:
            self._len    = len(f[split]["labels"])
            self._labels = f[split]["labels"][:]   # load labels into RAM (small)

    def __len__(self):
        return self._len

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")

    def __getitem__(self, idx):
        self._open()
        img   = self._h5[self._split]["images"][idx]   # (H, W, 3) uint8 RGB
        label = int(self._labels[idx])

        if self._transform:
            img = self._transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


def _build_transform(augment: bool, aug_cfg: list[dict] | None, size: int) -> A.Compose:
    ops: list = []
    if augment and aug_cfg:
        for entry in aug_cfg:
            name   = entry["name"]
            params = entry.get("params", {})
            if name == "Normalize":
                continue   # handled last
            cls = getattr(A, name, None)
            if cls is None:
                print(f"  WARN: albumentations.{name} not found — skipping")
                continue
            ops.append(cls(**params))
    ops += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(ops)


# ── Training helpers ──────────────────────────────────────────────────────────

def _class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weight for BCEWithLogitsLoss pos_weight."""
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    if n_real == 0 or n_fake == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(n_real / n_fake, device=device)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(imgs)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

    mean_loss = total_loss / max(len(all_labels), 1)
    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.5
    return mean_loss, auc


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficientnet_b4_mvp.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    m_cfg    = cfg["model"]
    d_cfg    = cfg["data"]
    t_cfg    = cfg["training"]
    aug_cfg  = cfg.get("augmentation", [])
    log_cfg  = cfg.get("logging", {})
    exp_cfg  = cfg.get("export", {})

    # ── device ───────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[train] Device: {device}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = (not args.no_wandb and
                 "WANDB_PROJECT" in os.environ and
                 log_cfg.get("wandb_project"))
    if use_wandb:
        import wandb
        wandb.init(project=log_cfg["wandb_project"], config=cfg)
        print(f"[train] W&B logging → {log_cfg['wandb_project']}")
    else:
        print("[train] Logging to stdout only (set WANDB_PROJECT env var to enable W&B)")

    # ── datasets ─────────────────────────────────────────────────────────────
    h5_path  = d_cfg["dataset_h5"]
    size     = d_cfg["image_size"]
    train_tf = _build_transform(augment=True,  aug_cfg=aug_cfg, size=size)
    val_tf   = _build_transform(augment=False, aug_cfg=None,    size=size)

    train_ds = FaceDataset(h5_path, "train", train_tf)
    val_ds   = FaceDataset(h5_path, "val",   val_tf)

    nw = min(t_cfg.get("num_workers", 4), 4)
    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"],
                              shuffle=True,  num_workers=nw, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"],
                              shuffle=False, num_workers=nw, pin_memory=False)

    print(f"[train] Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = timm.create_model(
        m_cfg["name"],
        pretrained=m_cfg["pretrained"],
        num_classes=m_cfg["num_classes"],
        drop_rate=m_cfg.get("drop_rate", 0.3),
        drop_path_rate=m_cfg.get("drop_path_rate", 0.2),
    )
    model = model.to(device)
    print(f"[train] Model: {m_cfg['name']}  params: {sum(p.numel() for p in model.parameters()):,}")

    # ── loss — use pos_weight if class imbalance ──────────────────────────────
    train_labels = train_ds._labels
    if t_cfg.get("auto_class_weight", True):
        pos_weight = _class_weights(train_labels, device)
        print(f"[train] pos_weight (fake/real balance): {pos_weight.item():.3f}")
    else:
        pos_weight = torch.tensor(1.0, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=t_cfg["lr"],
                                  weight_decay=t_cfg.get("weight_decay", 1e-4))
    total_steps = t_cfg["epochs"] * len(train_loader)
    warmup_steps = t_cfg.get("warmup_epochs", 1) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)

    # ── checkpoint dir ───────────────────────────────────────────────────────
    ckpt_path = Path(t_cfg.get("checkpoint_path", "checkpoints/best.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_auc    = 0.0
    history     = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        print(f"[train] Resumed from {args.resume} (epoch {start_epoch}, best AUC {best_auc:.4f})")

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\n[train] Starting {t_cfg['epochs']} epochs…\n")
    for epoch in range(start_epoch, t_cfg["epochs"]):
        t0 = time.time()

        tr_loss, tr_auc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_auc = _run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = dict(epoch=epoch, tr_loss=tr_loss, tr_auc=tr_auc,
                   va_loss=va_loss, va_auc=va_auc, lr=lr_now, elapsed=elapsed)
        history.append(row)

        marker = " ← best" if va_auc > best_auc else ""
        print(f"Epoch {epoch+1:02d}/{t_cfg['epochs']}  "
              f"tr_loss={tr_loss:.4f} tr_auc={tr_auc:.4f}  "
              f"va_loss={va_loss:.4f} va_auc={va_auc:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s{marker}")

        if use_wandb:
            import wandb
            wandb.log(row)

        # Save best checkpoint
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_auc":   best_auc,
                "config":     cfg,
            }, str(ckpt_path))
            print(f"  ✓ Checkpoint saved → {ckpt_path}  (val AUC {best_auc:.4f})")

    # ── save training history ─────────────────────────────────────────────────
    hist_path = ckpt_path.parent / "history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"\n[train] Best val AUC: {best_auc:.4f}")
    print(f"[train] History     : {hist_path}")
    print(f"[train] Checkpoint  : {ckpt_path}")
    print(f"\nNext: python scripts/export_onnx.py --checkpoint {ckpt_path} --version v0.1")

    if use_wandb:
        import wandb
        wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
