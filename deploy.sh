#!/bin/bash

# AI Grid Model Deployment Script
set -e

echo "Starting AI Grid Model Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Build the frontend
print_status "Building React frontend..."
cd frontend
npm install
npm run build
cd ..

# Build and start services with Docker Compose
print_status "Starting services with Docker Compose..."
docker-compose down 2>/dev/null || true
docker-compose up --build -d

# Wait for services to be healthy
print_status "Waiting for services to start..."
sleep 10

# Check backend health
print_status "Checking backend health..."
if curl -f http://localhost:5001/api/health &>/dev/null; then
    print_status "Backend is healthy"
else
    print_error "❌ Backend health check failed"
    docker-compose logs backend
    exit 1
fi

# Check frontend
print_status "Checking frontend..."
if curl -f http://localhost:3000 &>/dev/null; then
    print_status "Frontend is responding"
else
    print_error "❌ Frontend check failed"
    docker-compose logs frontend
    exit 1
fi

print_status "Deployment completed successfully!"
echo ""
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:5001"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"
echo "To restart: docker-compose restart"
