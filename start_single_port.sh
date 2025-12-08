#!/bin/bash
# Start both backend and frontend servers for single-port deployment
# Backend runs on port 5001, Frontend on port 60000 with API proxying

echo "Starting AI Grid Demo with single-port architecture..."
echo "Backend: http://localhost:5001 (internal)"
echo "Frontend: http://localhost:60000 (public - all API calls proxied)"

# Activate virtual environment
source venv/bin/activate

# Start backend in background
echo "Starting backend server..."
python scripts/start_api.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend server
echo "Starting frontend server with API proxy..."
cd frontend
npm run serve &
FRONTEND_PID=$!

echo ""
echo "✅ Servers started successfully!"
echo "🌐 Access the application at: http://localhost:60000"
echo "🔧 Backend PID: $BACKEND_PID"
echo "⚛️  Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to kill both processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $FRONTEND_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    echo "✅ All servers stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for background processes
wait
