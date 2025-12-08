#!/bin/bash

# Complete server startup script for AI Grid Model
# This script starts both backend and frontend servers

echo "Starting AI Grid Model Servers"
echo "=================================="

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Start backend server in background
print_status "Starting backend API server on port 5001..."
python scripts/start_api.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -f http://localhost:5001/api/health &>/dev/null; then
    print_status "Backend server started successfully"
    print_status "   API available at: http://localhost:5001"
else
    print_error "❌ Backend server failed to start"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend server in background
print_status "Starting frontend React server..."
cd frontend
npm start &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

# Check if frontend is running (on port 60000 as configured)
if curl -f http://localhost:60000 &>/dev/null; then
    print_status "Frontend server started successfully"
    print_status "   Web interface at: http://localhost:60000"
else
    print_warning "Frontend server may still be starting..."
    print_status "   Web interface should be at: http://localhost:60000"
fi

cd ..

print_status ""
print_status "Both servers are now running!"
print_status ""
print_status "Frontend (Web Interface): http://localhost:60000"
print_status "Backend API: http://localhost:5001"
print_status ""
print_status "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    print_status "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for user interrupt
wait
