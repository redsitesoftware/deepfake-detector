#!/bin/bash
# ============================================================
# 2_train.sh — Fine-tune EfficientNet-B4 on DF40 dataset
#
# Starts from the best saved checkpoint (Epoch 1, val AUC=0.538)
# and fine-tunes the full model at a very low learning rate (5e-6)
# to avoid the catastrophic overfit that happened in the first run.
#
# What happened in run 1:
#   - Epochs 1-5: backbone frozen, val AUC ~0.53 (plateau)
#   - Epoch 6: backbone unfroze at lr=1e-4 → overfit, val AUC collapsed to 0.09
#
# This run fixes that:
#   - Start from best checkpoint (epoch 1)
#   - No frozen phase (--freeze-backbone-epochs 0)
#   - lr=5e-6 (50x lower than what caused collapse)
#   - 15 epochs
#   - Web server must be OFF (don't run 4_webserver.sh first)
#
# Expected outcome:
#   - Epoch 1-3: val AUC should climb from 0.538 toward 0.65+
#   - Epoch 10-15: target AUC 0.75-0.85
#   - ETA: ~8-10 hours total (no competing processes)
#
# Usage:
#   cd /tmp/deepfake-detector
#   bash scripts/run/2_train.sh
#
# Log output goes to: /tmp/finetune_log2.txt
# Best checkpoint saved to: checkpoints/effnb4_finetuned_best.pt
# ============================================================

cd /tmp/deepfake-detector

PYTHON=/tmp/bridge-venv/bin/python
LOG=/tmp/finetune_log2.txt

echo "=== Starting training run 2 ==="
echo "    Checkpoint: checkpoints/effnb4_finetuned_best.pt (epoch 1, AUC=0.538)"
echo "    LR: 5e-6  Epochs: 15  Batch: 32"
echo "    Log: $LOG"
echo ""
echo "Watch progress with:"
echo "  tail -f $LOG | tr '\\r' '\\n' | grep -v 'batch/s'"
echo ""

nohup $PYTHON scripts/finetune_pretrained.py \
    --checkpoint checkpoints/effnb4_finetuned_best.pt \
    --df40-dir data/df40 \
    --epochs 15 \
    --freeze-backbone-epochs 0 \
    --batch-size 32 \
    --num-workers 0 \
    --lr 5e-6 \
    > "$LOG" 2>&1 &

echo "PID=$! — training running in background"
echo "To stop training: kill $!"
