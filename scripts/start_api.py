#!/usr/bin/env python3
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_grid_demo.api import app

if __name__ == "__main__":
    print("Starting AI Grid Demo API server on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
