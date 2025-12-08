# Classical Mathematical Models for AI Grid Comparison

This implementation provides advanced classical mathematical approaches for each AI Grid use case, enabling direct comparison with neural network models.

## 🎯 **Implementation Overview**

### **Classical Models Implemented**

| Use Case | Classical Model | Key Algorithm | Comparison Focus |
|----------|----------------|----------------|------------------|
| **Load Forecasting** | Multivariate VAR | Statsmodels VAR(p) | Temporal patterns vs NN learning |
| **State Estimation** | DC WLS Estimator | Matrix inversion | Measurement noise handling |
| **Congestion Prediction** | PTDF + Logistic | Scikit-learn LogisticRegression | Linearity vs nonlinearity |
| **OPF Surrogate** | DC-OPF Solver | CVXPY optimization | Exact vs approximate solutions |
| **Spatiotemporal** | PCA + VAR | Dimensionality reduction + VAR | Structure vs flexibility |
| **Anomaly Detection** | χ² Residual Test | Statistical testing | Model-based vs data-driven |

## 🏗️ **Architecture**

```
├── ai_grid_demo/
│   ├── classical_models.py          # Core classical implementations
│   ├── api.py                      # REST endpoints for classical models
│   └── ...
├── frontend/src/components/
│   ├── UseCaseDemo.tsx             # Updated with classical model buttons
│   ├── NeuralNetworkVisualizer.tsx # NN architecture visualization
│   └── ...
└── requirements.txt                # Added dependencies (cvxpy, statsmodels, etc.)
```

## 📊 **Model Specifications**

### **1. Load Forecasting - Multivariate VAR**

**Mathematical Model:**
```
𝐋_t = c + Σ_{k=1 to p} A_k * 𝐋_{t-k} + ε_t
```

**Implementation:**
- **Library**: `statsmodels.tsa.api.VAR`
- **Lag Selection**: AIC-based automatic selection
- **Data**: 70% train / 30% test split
- **Metrics**: MSE, MAE, MAPE

**Key Features:**
- Captures cross-load correlations
- Automatic lag order selection
- Stationarity handling

### **2. State Estimation - DC WLS**

**Mathematical Model:**
```
z_t = H * x_t + e_t,    e_t ~ N(0, R)
x̂_t = (H^T R⁻¹ H)⁻¹ H^T R⁻¹ z_t
```

**Implementation:**
- **Measurements**: Synthetic voltage/current with noise
- **Jacobian**: Simplified DC power flow equations
- **Solver**: Direct matrix inversion
- **Metrics**: MSE, MAE, max error

**Key Features:**
- Optimal under Gaussian assumptions
- Fast computation (O(n³) for n state variables)
- Exact confidence intervals

### **3. Congestion Prediction - PTDF + Logistic**

**Mathematical Model:**
```
F_line ≈ PTDF[line, :] @ P_injection
P(congested | features) = σ(w^T x + b)
```

**Features per line:**
- DC flow approximation
- Relative loading (% of capacity)
- Local injection statistics

**Implementation:**
- **PTDF**: Precomputed power transfer factors
- **Classifier**: Scikit-learn LogisticRegression
- **Threshold**: 90% capacity for congestion
- **Metrics**: AUC, precision, recall, F1

### **4. OPF Surrogate - DC-OPF Solver**

**Mathematical Model:**
```
min Σ_g (a_g P_g² + b_g P_g + c_g)
s.t. power balance, generator limits, line limits
```

**Implementation:**
- **Solver**: CVXPY with OSQP
- **Variables**: Generator setpoints + bus angles
- **Constraints**: DC power flow equations
- **Metrics**: Solution quality, computation time

**Key Features:**
- Exact DC solution as baseline
- Quadratic programming formulation
- Feasibility guarantees

### **5. Spatiotemporal - PCA + VAR**

**Mathematical Model:**
```
y_t ≈ C z_t + v_t    (PCA observation)
z_{t+1} = A z_t + w_t  (VAR state transition)
```

**Implementation:**
- **PCA**: Scikit-learn with explained variance selection
- **VAR**: Statsmodels on latent space
- **Forecasting**: Multi-step ahead prediction
- **Metrics**: MSE, MAE, correlation

### **6. Anomaly Detection - χ² Test**

**Mathematical Model:**
```
r_t = z_t - H x̂_t    (residuals)
J_t = r_t^T R⁻¹ r_t  (χ² statistic)
anomaly if J_t > χ²_{α, dof}
```

**Implementation:**
- **State Estimator**: Integrated WLS
- **Threshold**: χ² distribution quantile
- **Anomalies**: Synthetic spikes + line outages
- **Metrics**: AUC, precision, detection rate

## 🚀 **Usage**

### **Backend Testing**

```bash
# Start the backend
cd ai-grid-demo
source venv/bin/activate
python scripts/start_api.py

# Test classical models
curl http://localhost:5001/api/classical/load-forecasting
curl http://localhost:5001/api/classical/state-estimation
curl http://localhost:5001/api/classical/congestion-prediction
curl http://localhost:5001/api/classical/opf-solver
curl http://localhost:5001/api/classical/spatiotemporal
curl http://localhost:5001/api/classical/anomaly-detection
```

### **Frontend Testing**

```bash
# Start frontend
cd frontend
npm start

# Navigate to: Use Cases → Select any use case
# Click both "🧠 Neural Network" and "📊 Classical Model" buttons
# View comparative results below
```

## 📈 **Comparison Framework**

### **Metrics Comparison**

| Use Case | Primary Metric | Secondary Metrics |
|----------|----------------|-------------------|
| Forecasting | MAPE | MSE, MAE, correlation |
| State Estimation | MAE | MSE, max error |
| Congestion | AUC | Precision, recall, F1 |
| OPF | Cost gap | Feasibility rate, solve time |
| Spatiotemporal | MSE | MAE, explained variance |
| Anomaly | AUC | Precision, false positive rate |

### **Expected Results**

**Classical vs Neural Networks:**

1. **Forecasting**: VAR often better on short horizons, NN better on complex patterns
2. **State Estimation**: WLS optimal under assumptions, NN handles model mismatch
3. **Congestion**: PTDF+logistic interpretable, NN learns nonlinear relationships
4. **OPF**: DC-OPF exact baseline, NN surrogate enables real-time use
5. **Spatiotemporal**: PCA+VAR structured, NN more flexible
6. **Anomaly**: χ² statistical guarantees, NN learns complex patterns

## 🔧 **Technical Implementation**

### **Dependencies Added**

```txt
cvxpy>=1.3           # Convex optimization for OPF
statsmodels>=0.13    # VAR models and statistical tests
scikit-learn>=1.3    # Logistic regression, PCA
scipy>=1.9          # Sparse matrices, statistical functions
```

### **API Endpoints**

```python
# Classical model endpoints
@app.route('/api/classical/load-forecasting', methods=['POST'])
@app.route('/api/classical/state-estimation', methods=['POST'])
@app.route('/api/classical/congestion-prediction', methods=['POST'])
@app.route('/api/classical/opf-solver', methods=['POST'])
@app.route('/api/classical/spatiotemporal', methods=['POST'])
@app.route('/api/classical/anomaly-detection', methods=['POST'])
```

### **Data Processing**

- **Training Split**: First 70% of time series
- **Normalization**: StandardScaler for VAR models
- **Validation**: Holdout testing on remaining 30%
- **Anomaly Injection**: Synthetic anomalies for evaluation

## 🎨 **Frontend Features**

### **Dual Model Testing**
- Side-by-side neural network and classical model buttons
- Separate result displays for each approach
- Comparative analysis table showing performance differences

### **Result Visualization**
- **Forecasting**: Time series plots with confidence intervals
- **State Estimation**: Error distributions and scatter plots
- **Congestion**: ROC curves and confusion matrices
- **OPF**: Cost comparisons and feasibility indicators
- **Spatiotemporal**: Multi-step ahead predictions
- **Anomaly**: Detection performance and threshold analysis

## 🔬 **Research Applications**

### **Ablation Studies**
- Compare classical baselines with NN improvements
- Quantify benefits of deep learning approaches
- Identify where traditional methods still excel

### **Interpretability Research**
- Classical models provide ground truth for comparison
- Statistical guarantees vs black-box performance
- Trade-offs between accuracy and interpretability

### **Real-world Validation**
- Classical methods validate NN performance
- Hybrid approaches combining both paradigms
- Deployment confidence through rigorous comparison

## 📚 **References**

### **Key Papers**
- **VAR Forecasting**: Lütkepohl (2005) "New Introduction to Multiple Time Series Analysis"
- **State Estimation**: Abur & Exposito (2004) "Power System State Estimation"
- **PTDF Methods**: Christie et al. (2000) "Transmission Management in the Deregulated Environment"
- **DC-OPF**: Wood & Wollenberg (1996) "Power Generation, Operation, and Control"
- **State-Space Models**: Shumway & Stoffer (2017) "Time Series Analysis and Its Applications"
- **χ² Tests**: Barnett & Lewis (1994) "Outliers in Statistical Data"

### **Software Libraries**
- **CVXPY**: Diamond & Boyd (2016) "CVXPY: A Python-embedded modeling language"
- **Statsmodels**: Seabold & Perktold (2010) "Statsmodels: Econometric and statistical modeling"
- **Scikit-learn**: Pedregosa et al. (2011) "Scikit-learn: Machine Learning in Python"

This implementation provides a comprehensive classical baseline suite for rigorous evaluation of neural network approaches in power grid applications! 🚀
