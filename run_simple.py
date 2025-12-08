#!/usr/bin/env python3
"""
Super simple API runner - no import issues
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set environment
os.environ.setdefault('FLASK_ENV', 'production')

try:
    from ai_grid_demo.api import app
    print("API loaded successfully!")
    print("Starting on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
except Exception as e:
    print(f"Error: {e}")
    print("Troubleshooting:")
    print(f"- Current dir: {current_dir}")
    print(f"- Python path: {sys.path[:3]}...")
    print("- Make sure you're running from project root")
