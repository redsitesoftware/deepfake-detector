#!/bin/bash
# ============================================================
# 1_fix_venv.sh — Repair the bridge venv after PySide6 broke deps
#
# Run this FIRST before anything else.
# PySide6 installation corrupted numpy, h5py, anyio, efficientnet_pytorch.
# This script reinstalls everything needed.
#
# Usage:
#   cd /tmp/deepfake-detector
#   bash scripts/run/1_fix_venv.sh
# ============================================================

set -e
PYTHON=/tmp/bridge-venv/bin/python

echo "=== Fixing bridge venv dependencies ==="

$PYTHON -m pip install --force-reinstall \
    "numpy<2" \
    h5py \
    efficientnet_pytorch \
    scikit-learn \
    tqdm \
    albumentations \
    "anyio>=4.4" \
    PySide6

echo ""
echo "=== Verifying imports ==="
$PYTHON -c "
from efficientnet_pytorch import EfficientNet
import h5py, sklearn, tqdm, albumentations, anyio, PySide6
import numpy as np
print(f'  numpy:                {np.__version__}')
print(f'  efficientnet_pytorch: OK')
print(f'  h5py:                 OK')
print(f'  PySide6:              {PySide6.__version__}')
print(f'  anyio:                OK')
print('')
print('All dependencies OK — ready to train.')
"
