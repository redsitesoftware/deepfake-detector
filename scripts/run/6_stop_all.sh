#!/bin/bash
# ============================================================
# 6_stop_all.sh — Kill all deepfake-detector background processes
#
# Kills: uvicorn web server, training run, any stray Python processes
# that are associated with this project.
#
# Usage:
#   bash scripts/run/6_stop_all.sh
# ============================================================

echo "=== Stopping all deepfake-detector processes ==="

# Kill by log file ownership
for LOG in /tmp/finetune_log2.txt /tmp/dd_server.log; do
    PIDS=$(lsof -t "$LOG" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        for PID in $PIDS; do
            CMD=$(ps -p "$PID" -o comm= 2>/dev/null)
            echo "  Killing PID $PID ($CMD) — had $LOG open"
            kill "$PID" 2>/dev/null
        done
    fi
done

# Also grep for uvicorn on port 8765
PORT_PID=$(lsof -ti :8765 2>/dev/null)
if [ -n "$PORT_PID" ]; then
    echo "  Killing PID $PORT_PID (port 8765)"
    kill "$PORT_PID" 2>/dev/null
fi

sleep 2

# Confirm
STILL_RUNNING=$(lsof -ti :8765 2>/dev/null)
if [ -z "$STILL_RUNNING" ]; then
    echo ""
    echo "All processes stopped."
else
    echo ""
    echo "Port 8765 still in use — force killing..."
    kill -9 "$STILL_RUNNING" 2>/dev/null
fi
