#!/bin/bash
# ============================================================
# 3_export_onnx.sh — Export fine-tuned checkpoint to ONNX
#
# Run this AFTER training completes and val AUC > 0.75.
# Exports to both fp32 and fp16 ONNX formats.
# The web UI and demo.py automatically pick up the model
# from models/efficientnet_b4_deepfake_v1.0.onnx
#
# Usage:
#   cd /tmp/deepfake-detector
#   bash scripts/run/3_export_onnx.sh
# ============================================================

cd /tmp/deepfake-detector

PYTHON=/tmp/bridge-venv/bin/python
CHECKPOINT=checkpoints/effnb4_finetuned_best.pt

echo "=== Exporting ONNX model ==="
echo "    Source:  $CHECKPOINT"
echo "    Output:  models/efficientnet_b4_deepfake_v1.0.onnx"
echo ""

$PYTHON scripts/export_onnx.py \
    --checkpoint "$CHECKPOINT" \
    --model-type dfbench \
    --version v1.0

echo ""
echo "=== Verifying ONNX model ==="
$PYTHON -c "
import onnxruntime as ort
import numpy as np

model_path = 'models/efficientnet_b4_deepfake_v1.0.onnx'
sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
dummy = np.random.randn(1, 3, 380, 380).astype(np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: dummy})
import scipy.special
probs = scipy.special.softmax(out[0], axis=1)
print(f'  Model loaded OK')
print(f'  Output shape: {out[0].shape}')
print(f'  P(fake) on dummy input: {probs[0][1]:.4f}')
print('')
print('ONNX export verified — web UI will auto-load this model on next server start.')
"
