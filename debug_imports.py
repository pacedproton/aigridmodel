#!/usr/bin/env python3
"""
Debug script to test imports on deployment host
"""
import sys
import os

print("=== DEBUGGING IMPORTS ===")
print(f"Current directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

print("\n=== PYTHON PATH ===")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

print("\n=== CHECKING DIRECTORY STRUCTURE ===")
if os.path.exists('ai_grid_demo'):
    print("✓ ai_grid_demo directory exists")
    if os.path.exists('ai_grid_demo/__init__.py'):
        print("✓ ai_grid_demo/__init__.py exists")
    if os.path.exists('ai_grid_demo/data'):
        print("✓ ai_grid_demo/data directory exists")
        if os.path.exists('ai_grid_demo/data/__init__.py'):
            print("✓ ai_grid_demo/data/__init__.py exists")
        if os.path.exists('ai_grid_demo/data/simulator.py'):
            print("✓ ai_grid_demo/data/simulator.py exists")
else:
    print("✗ ai_grid_demo directory NOT found")

print("\n=== TESTING IMPORTS ===")
try:
    sys.path.insert(0, os.getcwd())
    print(f"Added {os.getcwd()} to path")

    # Test basic import
    import ai_grid_demo
    print("✓ ai_grid_demo import successful")

    # Test data module import
    from ai_grid_demo.data.simulator import simulate_grid_timeseries
    print("✓ ai_grid_demo.data.simulator import successful")

    # Test API import
    from ai_grid_demo.api import app
    print("✓ ai_grid_demo.api import successful")

    print("\n🎉 ALL IMPORTS WORK! The issue might be elsewhere.")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("This indicates a Python path or package structure issue.")

print("\n=== SUGGESTED FIXES ===")
print("1. Make sure you're in the project root directory")
print("2. Try: PYTHONPATH=. python scripts/start_api.py")
print("3. Or: export PYTHONPATH=. && python scripts/start_api.py")
print("4. Check if virtual environment is activated")

