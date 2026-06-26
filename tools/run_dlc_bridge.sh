#!/bin/bash
# run_dlc_bridge.sh — launcher using the bridge venv
exec /tmp/bridge-venv/bin/python /tmp/deepfake-detector/tools/dlc_bridge.py "$@"
