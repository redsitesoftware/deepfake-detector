#!/bin/bash
# ============================================================
# 4_webserver.sh — Start the deepfake detector web UI
#
# ⚠️  DO NOT run while training (2_train.sh) is active.
#     Both use MPS — competing processes cause 73x slowdown.
#
# Opens: http://localhost:8765
#
# What the web UI does:
#   1. Upload a face image (e.g. a celebrity photo)
#   2. Your webcam streams to the server at 15fps
#   3. Server runs Deep-Live-Cam face swap on each frame
#   4. Server runs CNN + Temporal + Liveness detection on swapped frame
#   5. Returns swapped frame + scores to browser
#
# Panels:
#   Left:   Your raw webcam (with face bbox)
#   Middle: DLC face-swapped output
#   Right:  Detection overlay (toggle with D key or ⊞ button)
#
# Controls:
#   Space   Reset temporal baseline (clears detection history)
#   D       Toggle triple/dual view
#   ↩       Change Face (re-upload source image)
#
# Usage:
#   cd /tmp/deepfake-detector
#   bash scripts/run/4_webserver.sh
# ============================================================

cd /tmp/deepfake-detector

LOG=/tmp/dd_server.log

echo "=== Starting deepfake detector web server ==="
echo "    URL:  http://localhost:8765"
echo "    Log:  $LOG"
echo ""

nohup /tmp/bridge-venv/bin/python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8765 \
    --reload \
    --reload-dir api \
    --reload-dir deepfake_detector \
    >> "$LOG" 2>&1 &

echo "PID=$!"
echo ""
echo "Waiting for server to start..."
sleep 8

STATUS=$(curl -s http://localhost:8765/health 2>/dev/null)
if echo "$STATUS" | grep -q '"status":"ok"'; then
    echo "Server is up!"
    open http://localhost:8765
else
    echo "Server may still be starting. Check: tail -f $LOG"
fi
