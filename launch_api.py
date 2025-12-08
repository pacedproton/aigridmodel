#!/usr/bin/env python3
"""
API Launcher - Run this from the project root directory
"""
import sys
import os

# Ensure we're in the project root
project_root = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(project_root, 'ai_grid_demo')):
    print("Error: Please run this script from the project root directory")
    sys.exit(1)

# Add project root to Python path
sys.path.insert(0, project_root)

# Change to project root directory
os.chdir(project_root)

# Now import and run the app
try:
    from ai_grid_demo.api import app
    print("Starting AI Grid Demo API server on http://localhost:5001")
    print("Press Ctrl+C to stop")
    app.run(debug=False, host='0.0.0.0', port=5001)
except ImportError as e:
    print(f"Import error: {e}")
    print("Current directory:", os.getcwd())
    print("Python path:", sys.path)
    sys.exit(1)

