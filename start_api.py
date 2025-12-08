#!/usr/bin/env python3
"""
Universal API launcher that works from any directory
"""
import sys
import os

# Find the project root (directory containing ai_grid_demo)
current_dir = os.getcwd()
project_root = None

# Check current directory first
if os.path.exists(os.path.join(current_dir, 'ai_grid_demo')):
    project_root = current_dir
else:
    # Search parent directories
    parent_dir = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(parent_dir, 'ai_grid_demo')):
        project_root = parent_dir

# If still not found, try to find it relative to this script
if project_root is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.dirname(script_dir)  # scripts/ -> project root
    if os.path.exists(os.path.join(candidate, 'ai_grid_demo')):
        project_root = candidate

if project_root is None:
    print("ERROR: Cannot find ai_grid_demo module. Please run from project root directory.")
    print(f"Current directory: {current_dir}")
    print("Make sure you're in the directory containing 'ai_grid_demo' folder.")
    sys.exit(1)

# Add project root to Python path
sys.path.insert(0, project_root)

# Change to project root to ensure relative imports work
os.chdir(project_root)

print(f"Starting AI Grid Demo API from: {project_root}")

try:
    from ai_grid_demo.api import app
    print("API module loaded successfully")
    print("Starting Flask server on http://localhost:5001")
    print("Press Ctrl+C to stop")
    app.run(debug=False, host='0.0.0.0', port=5001)
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print("Available directories:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  {item}/")
    sys.exit(1)

