# AI Grid Model - Deployment Guide

This guide covers how to deploy both the React frontend and Flask backend together.

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Build and deploy with Docker Compose
./deploy.sh
```

This will:
- Build the React frontend
- Create Docker containers for both frontend and backend
- Start services on ports 3000 (frontend) and 5001 (backend)

### Option 2: Development Mode

```bash
# Run both frontend and backend locally
./dev.sh
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   React App     │    │   Flask API     │
│   (Port 3000)   │◄──►│   (Port 5001)   │
│                 │    │                 │
│ • Dashboard     │    │ • Data Generation│
│ • Model Training│    │ • ML Predictions│
│ • Data Viz      │    │ • Network Plots │
└─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Docker & Docker Compose
- Node.js 16+ (for development)
- Python 3.11+ (for development)

## 🔧 Configuration

### Environment Variables

Copy `config.example` to set up environment variables:

```bash
cp config.example .env
```

Key configurations:
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:5001)
- `FLASK_ENV`: Flask environment (development/production)

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Start services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Commands

```bash
# Build backend
docker build -f Dockerfile.backend -t ai-grid-backend .

# Build frontend
cd frontend
docker build -t ai-grid-frontend .
cd ..

# Run services
docker run -d -p 5001:5001 --name backend ai-grid-backend
docker run -d -p 3000:80 --name frontend --link backend ai-grid-frontend
```

## 🌐 Production Deployment

### 1. Cloud Platforms

**Railway:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Render:**
- Connect GitHub repo
- Set build command: `./deploy.sh`
- Set start command: `docker-compose up -d`

**AWS/GCP/Azure:**
```bash
# Build and push to container registry
docker tag ai-grid-backend your-registry/backend:latest
docker tag ai-grid-frontend your-registry/frontend:latest
docker push your-registry/backend:latest
docker push your-registry/frontend:latest
```

### 2. Traditional Hosting

**Backend (Flask):**
- Use Gunicorn for production
- Set up reverse proxy (nginx)
- Configure environment variables

**Frontend (React):**
- Deploy built files to any static hosting
- Configure API URL for your backend

### 3. nginx Configuration

```nginx
# Frontend
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Backend API
server {
    listen 80;
    server_name api.your-domain.com;

    location / {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔍 Testing Deployment

### Health Checks

```bash
# Backend health
curl http://localhost:5001/api/health

# Frontend response
curl http://localhost:3000
```

### API Endpoints

```bash
# Generate sample data
curl -X POST http://localhost:5001/api/generate-data -H "Content-Type: application/json" -d '{"n_steps": 100}'

# Get data statistics
curl http://localhost:5001/api/data/stats

# Train model
curl -X POST http://localhost:5001/api/train/spatiotemporal -H "Content-Type: application/json" -d '{"epochs": 5}'
```

## 📊 Monitoring

### Logs

```bash
# Docker logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Application logs
docker exec -it ai-grid-backend tail -f /app/logs/app.log
```

### Health Monitoring

```bash
# Check service status
docker-compose ps

# Resource usage
docker stats
```

## 🚨 Troubleshooting

### Common Issues

**Frontend can't connect to backend:**
- Check `REACT_APP_API_URL` environment variable
- Ensure backend container is running: `docker-compose ps`
- Check network connectivity: `docker network ls`

**Backend not starting:**
- Check logs: `docker-compose logs backend`
- Verify Python dependencies: `docker exec -it ai-grid-backend pip list`
- Check data directories exist

**Build failures:**
- Clear Docker cache: `docker system prune -a`
- Rebuild without cache: `docker-compose build --no-cache`

### Port Conflicts

```bash
# Find process using port
lsof -i :3000
lsof -i :5001

# Kill process
kill -9 <PID>
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [React Deployment Guide](https://create-react-app.dev/docs/deployment/)
- [Flask Production Deployment](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- [nginx Reverse Proxy](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)

## 🔐 Security Considerations

- Use HTTPS in production
- Set secure environment variables
- Implement proper CORS policies
- Use Docker secrets for sensitive data
- Regularly update base images
