#!/bin/bash

# AI Grid Model Development Script
set -e

echo "Starting AI Grid Model in Development Mode..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_dev() {
    echo -e "${BLUE}[DEV]${NC} $1"
}

# Function to cleanup background processes
cleanup() {
    print_status "Cleaning up..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install/update backend dependencies
print_status "Installing backend dependencies..."
pip install -e .

# Start backend in background
print_dev "Starting Flask backend on port 5001..."
python scripts/start_api.py &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Check if backend is running
if curl -f http://localhost:5001/api/health &>/dev/null; then
    print_status "Backend started successfully"
else
    print_error "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend in background
print_dev "Starting React frontend on port 3000..."
cd frontend
npm install
REACT_APP_API_URL=http://localhost:5001 npm start &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

print_status "Development servers started!"
echo ""
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
wait
