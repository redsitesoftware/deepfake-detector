#!/bin/bash
# ============================================================
# 5_watch_training.sh — Monitor training progress in real-time
#
# Shows only the meaningful lines (epoch results, errors).
# Strips out the noisy tqdm progress bar lines.
#
# Usage:
#   bash scripts/run/5_watch_training.sh
#
# What to look for:
#   GOOD: val AUC increasing each epoch (target > 0.75)
#   BAD:  train AUC >> val AUC (overfitting)
#   BAD:  val AUC < 0.5 (inversion — reduce LR further)
# ============================================================

LOG=/tmp/finetune_log2.txt

if [ ! -f "$LOG" ]; then
    echo "No training log found at $LOG"
    echo "Start training first with: bash scripts/run/2_train.sh"
    exit 1
fi

echo "=== Training log: $LOG ==="
echo "=== Ctrl+C to stop watching ==="
echo ""

# Show completed epochs first
echo "--- Completed epochs ---"
cat "$LOG" | tr '\r' '\n' | grep "^Epoch"
echo ""
echo "--- Live output (Ctrl+C to exit) ---"

# Then follow new output, filtering out tqdm noise
tail -f "$LOG" | tr '\r' '\n' | grep -v "batch/s" | grep -v "^$"
