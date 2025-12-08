#!/usr/bin/env python3
"""
Run API as Python module
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

# Run as module
os.system("python -m ai_grid_demo.api")

