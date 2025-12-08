#!/bin/bash

# Quick activation script for the virtual environment
echo "🐍 Activating AI Grid Demo virtual environment..."
source "$(dirname "$0")/venv/bin/activate"
echo "✅ Environment activated!"
echo "You can now run Python scripts directly, e.g.:"
echo "  python scripts/start_api.py"
echo "  python scripts/generate_data.py"

