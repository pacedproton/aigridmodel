# AI-Powered Grid Modeling Demonstrator

A comprehensive AI-powered platform demonstrating advanced machine learning applications in power grid management and control. Features six core use cases with professional mathematical foundations, interactive visualizations, and comprehensive model comparisons.

## Key Features

- 6 Use Cases: Load forecasting, state estimation, congestion prediction, OPF surrogate, spatiotemporal fusion, and anomaly detection
- Data generation: pandapower
- Advanced Mathematical Models: Bayesian Structural Time Series, Extended Kalman Filter, MCMC Logistic Regression, Interior Point OPF, Gaussian Process PDE, and Hidden Markov Change Point detection
- Interactive GUI:, comparative visualizations, and live training progress, LaTeX mathe
- Three-Way Comparisons: Neural Networks vs Classical Models vs Advanced Mathematics for each use case
- Network Visualization: IEEE 14-bus system with authentic pandapower-generated topology diagrams
- Comprehensive Testing: Automated test coverage with real-time monitoring dashboard
- Production Ready: Docker deployment with nginx reverse proxy and production configurations

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Docker & Docker Compose** (recommended)

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ai-grid-demo

# Deploy with Docker Compose
./deploy.sh
```

Access the application at:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5001

### Option 2: Local Development

```bash
# Backend setup
cd ai-grid-demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/start_api.py &

# Frontend setup (new terminal)
cd frontend
npm install
npm start
```

Access at:

- **Frontend**: http://localhost:60000
- **Backend API**: http://localhost:5001

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   Flask Backend  │
│   (Port 3000)    │◄──►│   (Port 5001)    │
│                 │    │                 │
│ • Dashboard      │    │ • Data Generation │
│ • Model Training │    │ • AI Predictions  │
│ • 3-Way Comparison│   │ • Network Plots   │
│ • Test Coverage  │    │ • Mathematical    │
└─────────────────┘    └─────────────────┘
```

### Core Components

**Backend (`ai_grid_demo/`)**

- `advanced_models.py` - 6 Advanced mathematical models with proper equations
- `classical_models.py` - Traditional approaches (VAR, WLS, PTDF, DC-OPF)
- `api.py` - REST API with UI button endpoints for testing
- `config.py` - Configuration management

**Frontend (`frontend/src/`)**

- `UseCaseDemo.tsx` - Main interface with model execution
- `ForecastingComparison.tsx` - Dedicated comparison page
- `ModelComparisonVisualizer.tsx` - Interactive charts and metrics
- `MathEquation.tsx` - LaTeX mathematical rendering
- `IEEE14BusVisualization.tsx` - Network topology viewer

## Advanced AI Use Cases

IMPORTANT: Most "neural network" implementations are mock/demo responses. Only spatiotemporal fusion has actual neural network code.\_

_Actual implementations = trainable PyTorch models with real training loops. Mock implementations = hardcoded responses for UI demo._

### 1. Load Forecasting

**Neural**: _Mock implementation (no actual NN)_
**Classical**: Multivariate Vector Autoregression (VAR)
**Advanced**: Bayesian Structural Time Series (MCMC)

### 2. State Estimation

**Neural**: _Mock implementation (no actual NN)_
**Classical**: DC Weighted Least Squares (WLS)
**Advanced**: Extended Kalman Filter (Nonlinear)

### 3. Congestion Prediction

**Neural**: _Mock implementation (no actual NN)_
**Classical**: PTDF + Logistic Regression
**Advanced**: MCMC Logistic Regression (Bayesian)

### 4. OPF Surrogate

**Neural**: _Mock implementation (no actual NN)_
**Classical**: DC Optimal Power Flow (QP/LP)
**Advanced**: Interior Point Method

### 5. Spatiotemporal Fusion

**Neural**: GNN + Transformer (actual implementation)
**Classical**: PCA + VAR state space models
**Advanced**: Gaussian Process with PDE constraints

### 6. Anomaly Detection

**Neural**: _Mock implementation (no actual NN)_
**Classical**: χ² Residual testing on state estimator
**Advanced**: Hidden Markov Change Point detection

## Mathematical Foundations

Each advanced model features proper mathematical formulations rendered with LaTeX:

```latex
% Bayesian Structural Time Series
\begin{pmatrix} \mu_t \\ \delta_t \\ \tau_t \\ s_t^{(1)} \\ \vdots \\ s_t^{(S)} \end{pmatrix}
= \mathbf{T} \begin{pmatrix} \mu_{t-1} \\ \delta_{t-1} \\ \tau_{t-1} \\ s_{t-1}^{(1)} \\ \vdots \\ s_{t-1}^{(S)} \end{pmatrix} + \mathbf{R} \eta_t

% Extended Kalman Filter
\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k)
```

## Interactive Features

- Three-Way Model Comparison: Side-by-side performance analysis
- **📈 Live Training Progress**: Real-time convergence curves and metrics
- Network Topology: IEEE 14-bus system with authentic PandaPower diagrams
- Test Coverage Dashboard: Real-time testing and quality monitoring
- **📚 Mathematical Documentation**: Interactive equation tooltips and explanations

## Production Deployment

### Docker Compose (Recommended)

```bash
# Build and deploy
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Deployment

```bash
# Build images
docker build -f Dockerfile.backend -t ai-grid-backend .
docker build -f frontend/Dockerfile -t ai-grid-frontend .

# Run services
docker run -d -p 5001:5001 --name backend ai-grid-backend
docker run -d -p 3000:80 --name frontend ai-grid-frontend
```

### Cloud Deployment Options

- **Railway**: Connect GitHub repo, automatic builds
- **Render**: Docker-based deployment
- **AWS/GCP/Azure**: Container registry deployment
- **Vercel/Netlify**: Frontend static hosting + backend API

## 📚 API Documentation

### Core Endpoints

```bash
# Health check
GET /api/health

# Data generation
POST /api/generate-data

# Model training
POST /api/train/{model_type}

# Predictions
POST /api/predict/{model_type}

# UI button endpoints (for testing)
POST /api/ui/neural-network/{use_case}
POST /api/ui/classical-model/{use_case}
POST /api/ui/advanced-math/{use_case}
POST /api/ui/show-comparison/{use_case}
```

### Network Visualization

```bash
# Get topology plot
GET /api/network/plot
```

## Testing & Quality Assurance

### Automated Testing Suite

```bash
# Run comprehensive tests
python run_comprehensive_tests.py

# Run autonomous testing
python run_autonomous_tests.py

# Monitor test coverage
python monitor_servers.py
```

### Test Coverage Dashboard

Access via Navigation → "📈 Test Coverage"

- Real-time test execution monitoring
- Coverage metrics and trends
- Automated test scheduling
- Performance regression detection

## 🔧 Development

### Adding New Models

1. **Backend**: Add model class to `advanced_models.py` or `classical_models.py`
2. **API**: Create endpoint in `api.py`
3. **Frontend**: Add UI component in `components/`
4. **Math**: Update equations in `MathEquation.tsx`

### Project Structure

```
ai-grid-demo/
├── ai_grid_demo/          # Backend package
│   ├── advanced_models.py # Advanced mathematical models
│   ├── classical_models.py # Traditional approaches
│   ├── api.py            # Flask REST API
│   └── config.py         # Configuration
├── frontend/             # React application
│   ├── src/components/   # UI components
│   └── public/          # Static assets
├── scripts/             # Utility scripts
├── tests/               # Test suites
└── docker/              # Container configs
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
