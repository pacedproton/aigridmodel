#!/usr/bin/env python3
"""
Super simple API runner - runs as module
"""
import sys
import os

# Change to project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add current directory to Python path
sys.path.insert(0, script_dir)

# Set environment
os.environ.setdefault('FLASK_ENV', 'production')

try:
    # Import as module
    from ai_grid_demo.api import app
    print("API loaded successfully!")
    print("Starting on http://0.0.0.0:5001")
    print("Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=5001, debug=False)
except Exception as e:
    print(f"Error: {e}")
    print("Troubleshooting:")
    print(f"- Current dir: {os.getcwd()}")
    print(f"- Python path: {sys.path[:3]}...")
    print("- Make sure ai_grid_demo directory exists")
