#!/bin/bash
# API Launcher for deployment
cd "$(dirname "$0")"  # Go to script directory (project root)
export PYTHONPATH="$(pwd)"
echo "Starting from: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
python scripts/start_api.py
