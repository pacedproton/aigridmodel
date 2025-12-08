#!/bin/bash

echo "🔧 Setting up AI Grid Demo environment..."

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Setup frontend dependencies
echo "Setting up frontend..."
cd frontend
npm install
cd ..

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To start the demo:"
echo "  ./start_demo.sh"

