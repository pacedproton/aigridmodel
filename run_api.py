#!/usr/bin/env python3
"""
Direct API runner - run this from the project root directory
"""

from ai_grid_demo.api import app

if __name__ == "__main__":
    print("Starting AI Grid Demo API server on http://localhost:5001")
    print("Press Ctrl+C to stop")
    app.run(debug=True, host='0.0.0.0', port=5001)

