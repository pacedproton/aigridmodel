#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.getcwd())  # Add current directory to path

from ai_grid_demo.api import app

if __name__ == '__main__':
    print('Starting AI Grid Demo API server on http://localhost:5001')
    app.run(debug=False, host='0.0.0.0', port=5001)
