#!/bin/bash

echo "🌟 Starting AI Grid Demo - Streamlit Version..."
echo "This is a browser-based application with all functionality in one place"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$0")"

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Start Streamlit app
echo "🌐 Starting Streamlit application..."
echo ""
echo "📱 Access the application at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"

# Run Streamlit
streamlit run "$SCRIPT_DIR/streamlit_app.py" --server.headless true --server.port 8501

echo ""
echo "✅ Streamlit app stopped"

