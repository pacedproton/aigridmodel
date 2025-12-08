import React from "react";
import { MathJax, MathJaxContext } from "better-react-mathjax";

interface MathEquationProps {
  latex: string;
  inline?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export const MathEquation: React.FC<MathEquationProps> = ({
  latex,
  inline = false,
  className = "",
  style = {},
}) => {
  // Configure MathJax
  const mathJaxConfig = {
    tex: {
      inlineMath: [
        ["$", "$"],
        ["\\(", "\\)"],
      ],
      displayMath: [
        ["$$", "$$"],
        ["\\[", "\\]"],
      ],
      processEscapes: true,
      processEnvironments: true,
      packages: { "[+]": ["color"] },
    },
    options: {
      skipHtmlTags: ["script", "noscript", "style", "textarea", "pre"],
      ignoreHtmlClass: "tex2jax_ignore",
      processHtmlClass: "tex2jax_process",
    },
    loader: {
      load: ["[tex]/color"],
    },
  };

  // Format LaTeX for proper rendering
  const formattedLatex = inline ? `$${latex}$` : `$$${latex}$$`;

  return (
    <MathJaxContext config={mathJaxConfig}>
      <div
        className={`math-equation ${className}`}
        style={{
          fontFamily:
            '"Computer Modern", "Latin Modern Roman", "Times New Roman", serif',
          fontSize: inline ? "1em" : "1.2em",
          textAlign: inline ? "inherit" : "center",
          margin: inline ? "0 4px" : "1em 0",
          padding: "12px",
          backgroundColor: "#ffffff",
          border: "2px solid #e3f2fd",
          borderRadius: "8px",
          display: inline ? "inline-block" : "block",
          boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
          color: "#000000", // Ensure text is black
          minHeight: "24px",
          ...style,
        }}
      >
        <MathJax>{formattedLatex}</MathJax>
      </div>
    </MathJaxContext>
  );
};

// Predefined mathematical expressions for common use cases
export const MathExpressions = {
  // Forecasting - Bayesian Structural Time Series (Full MCMC Implementation)
  bayesianStructuralTimeSeries: {
    stateEvolution:
      "\\begin{pmatrix} \\mu_t \\\\ \\delta_t \\\\ \\tau_t \\\\ s_t^{(1)} \\\\ \\vdots \\\\ s_t^{(S)} \\end{pmatrix} = \\mathbf{T} \\begin{pmatrix} \\mu_{t-1} \\\\ \\delta_{t-1} \\\\ \\tau_{t-1} \\\\ s_{t-1}^{(1)} \\\\ \\vdots \\\\ s_{t-1}^{(S)} \\end{pmatrix} + \\mathbf{R} \\eta_t",
    observation:
      "y_t = \\mu_t + \\tau_t + \\sum_{j=1}^S s_t^{(j)} + \\beta^T x_t + \\epsilon_t",
    mcmc: "(\\theta^{(m+1)}, y^{(m+1)}) \\sim p(\\theta, y | y^*) \\propto p(y^* | \\theta, y) p(\\theta) p(y)",
    posterior: "p(\\theta | y^*) = \\int p(\\theta, y | y^*) dy",
  },

  // State Estimation - Extended Kalman Filter (Nonlinear)
  extendedKalmanFilter: {
    processModel:
      "\\mathbf{x}_k = f(\\mathbf{x}_{k-1}, \\mathbf{u}_k) + \\mathbf{w}_k, \\quad \\mathbf{w}_k \\sim \\mathcal{N}(0, \\mathbf{Q}_k)",
    measurementModel:
      "\\mathbf{z}_k = h(\\mathbf{x}_k) + \\mathbf{v}_k, \\quad \\mathbf{v}_k \\sim \\mathcal{N}(0, \\mathbf{R}_k)",
    jacobians:
      "\\mathbf{F}_k = \\frac{\\partial f}{\\partial \\mathbf{x}} \\bigg|_{\\hat{\\mathbf{x}}_{k-1|k-1}}, \\quad \\mathbf{H}_k = \\frac{\\partial h}{\\partial \\mathbf{x}} \\bigg|_{\\hat{\\mathbf{x}}_{k|k-1}}",
    update:
      "\\hat{\\mathbf{x}}_{k|k} = \\hat{\\mathbf{x}}_{k|k-1} + \\mathbf{K}_k (\\mathbf{z}_k - h(\\hat{\\mathbf{x}}_{k|k-1}))",
  },

  // Congestion Prediction - MCMC Logistic Regression
  mcmcLogistic: {
    fullPosterior:
      "p(\\beta, \\sigma^2 | \\mathbf{y}, \\mathbf{X}) \\propto p(\\mathbf{y} | \\beta, \\mathbf{X}, \\sigma^2) \\times p(\\beta | \\sigma^2) \\times p(\\sigma^2)",
    logisticLikelihood:
      "p(y_i | \\mathbf{x}_i, \\beta) = \\frac{\\exp(y_i \\cdot \\beta^T \\mathbf{x}_i)}{1 + \\exp(\\beta^T \\mathbf{x}_i)}",
    metropolisStep:
      "\\beta^{(t+1)} = \\beta^{(t)} + \\epsilon \\cdot \\nabla_\\beta \\log p(\\beta^{(t)} | \\mathbf{y}, \\mathbf{X}) + \\sqrt{2\\epsilon} \\cdot \\mathbf{Z}",
    acceptance:
      "\\alpha = \\min\\left(1, \\frac{p(\\beta^* | \\mathbf{y}, \\mathbf{X})}{p(\\beta^{(t)} | \\mathbf{y}, \\mathbf{X})} \\cdot \\frac{q(\\beta^{(t)} | \\beta^*)}{q(\\beta^* | \\beta^{(t)})}\\right)",
  },

  // OPF - Interior Point Method
  interiorPoint: {
    fullProblem:
      "\\min_{\\mathbf{x}} f(\\mathbf{x}) \\quad \\text{s.t.} \\quad \\mathbf{g}(\\mathbf{x}) \\leq 0, \\quad \\mathbf{h}(\\mathbf{x}) = 0, \\quad \\mathbf{x} \\in \\mathcal{X}",
    barrierForm:
      "\\min_{\\mathbf{x}, \\mathbf{y}, \\mathbf{z}} f(\\mathbf{x}) - \\mu \\sum_{i=1}^m \\log(-g_i(\\mathbf{x})) + \\frac{1}{2t} \\|\\mathbf{h}(\\mathbf{x})\\|^2",
    newtonStep:
      "\\begin{pmatrix} \\mathbf{x} \\\\ \\mathbf{y} \\\\ \\mathbf{z} \\end{pmatrix}^{(k+1)} = \\begin{pmatrix} \\mathbf{x} \\\\ \\mathbf{y} \\\\ \\mathbf{z} \\end{pmatrix}^{(k)} - \\mathbf{H}^{-1} \\nabla \\mathcal{L}",
    dualityGap:
      "\\eta = \\mathbf{y}^T \\mathbf{z}, \\quad \\mu_{k+1} = \\min(0.2, 10\\mu_k) \\cdot \\frac{\\eta}{m}",
  },

  // Spatiotemporal - Gaussian Process with PDE Constraints
  gaussianProcessPDE: {
    constrainedGP:
      "f(\\mathbf{s},t) | \\mathcal{L}f = 0 \\sim \\mathcal{GP}(m(\\mathbf{s},t), k((\\mathbf{s},t),((\\mathbf{s}',t')))",
    pdeKernel:
      "k((\\mathbf{s},t),((\\mathbf{s}',t')) = k_{\\text{spatial}}(\\mathbf{s},\\mathbf{s}') \\cdot k_{\\text{temporal}}(t,t') \\cdot c((\\mathbf{s},t),((\\mathbf{s}',t')))",
    greenKernel:
      "k_{\\text{Green}}((\\mathbf{s},t),((\\mathbf{s}',t')) = \\int \\phi(\\mathbf{s}-\\mathbf{r}) \\phi(\\mathbf{s}'-\\mathbf{r}) G(t,t';\\mathbf{r}) d\\mathbf{r}",
    posterior:
      "p(f | \\mathcal{D}, \\mathcal{L}f = 0) = \\mathcal{N}(\\mu_{\\text{post}}, \\Sigma_{\\text{post}})",
  },

  // Anomaly Detection - Hidden Markov Model with Change Points
  hiddenMarkovChangePoint: {
    fullLikelihood:
      "p(Y_{1:T}, S_{1:K} | \\theta) = p(S_{1:K}) \\prod_{k=1}^K p(Y_{T_{k-1}+1:T_k} | \\theta_k) \\cdot p(T_{1:K} | S_{1:K})",
    regimeModel:
      "Y_t | S_t = k \\sim \\mathcal{N}(\\mu_k, \\Sigma_k), \\quad S_t | S_{t-1} \\sim \\text{Markov chain}",
    changePointPosterior:
      "p(S_{1:T} | Y_{1:T}) \\propto p(Y_{1:T} | S_{1:T}) \\cdot p(S_{1:T})",
    anomalyScore:
      "A_t = 1 - \\max_k p(S_t = k | Y_{1:T}) \\cdot \\mathbb{I}\\{p(S_t = k | Y_{1:T}) > \\tau\\}",
  },

  // General mathematical notation
  general: {
    expectation: "\\mathbb{E}[X] = \\int x f(x) dx",
    variance: "\\text{Var}(X) = \\mathbb{E}[(X - \\mu)^2]",
    normal: "\\mathcal{N}(\\mu, \\sigma^2)",
    matrix:
      "\\mathbf{A} = \\begin{pmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{pmatrix}",
    gradient:
      "\\nabla f = \\begin{pmatrix} \\frac{\\partial f}{\\partial x_1} \\\\ \\vdots \\\\ \\frac{\\partial f}{\\partial x_n} \\end{pmatrix}",
    jacobian:
      "J = \\frac{\\partial f}{\\partial x} = \\begin{pmatrix} \\frac{\\partial f_1}{\\partial x_1} & \\cdots & \\frac{\\partial f_1}{\\partial x_n} \\\\ \\vdots & \\ddots & \\vdots \\\\ \\frac{\\partial f_m}{\\partial x_1} & \\cdots & \\frac{\\partial f_m}{\\partial x_n} \\end{pmatrix}",
  },
};

// Test component for demonstrating proper mathematical foundations
export const LaTeXTest: React.FC = () => {
  return (
    <div style={{ padding: "20px", backgroundColor: "#f5f5f5" }}>
      <h3>
        Advanced Mathematical Foundations - Now Showing Proper Complexity!
      </h3>

      <h4>🚫 BEFORE (Too Simple):</h4>
      <MathEquation latex="y_t = \mu_t + \tau_t + s_t + \epsilon_t" />
      <p style={{ color: "red" }}>
        ❌ This simple additive model doesn't reflect the complexity of advanced
        mathematical models!
      </p>

      <h4>NOW (Properly Complex):</h4>

      <h5>Bayesian Structural Time Series (Advanced MCMC):</h5>
      <MathEquation
        latex={MathExpressions.bayesianStructuralTimeSeries.stateEvolution}
      />
      <p>
        Full state space model with MCMC sampling - not just simple addition!
      </p>

      <h5>Extended Kalman Filter (Nonlinear State Estimation):</h5>
      <MathEquation latex={MathExpressions.extendedKalmanFilter.processModel} />
      <p>
        Nonlinear process and measurement models with Jacobian linearization!
      </p>

      <h5>MCMC Logistic Regression (Bayesian Classification):</h5>
      <MathEquation latex={MathExpressions.mcmcLogistic.fullPosterior} />
      <p>Full Bayesian posterior with Metropolis-Hastings sampling!</p>

      <h5>Interior Point OPF (Advanced Optimization):</h5>
      <MathEquation latex={MathExpressions.interiorPoint.fullProblem} />
      <p>Complete constrained optimization with barrier methods!</p>

      <h5>🌊 Gaussian Process PDE (Physics-Informed ML):</h5>
      <MathEquation latex={MathExpressions.gaussianProcessPDE.constrainedGP} />
      <p>Gaussian Processes with physical PDE constraints!</p>

      <h5>Hidden Markov Change Point (Regime Detection):</h5>
      <MathEquation
        latex={MathExpressions.hiddenMarkovChangePoint.fullLikelihood}
      />
      <p>Complete HMM likelihood with change point segmentation!</p>
    </div>
  );
};

// Component for displaying model equations with explanations
interface ModelEquationsProps {
  modelType:
    | "forecasting"
    | "state_estimation"
    | "congestion"
    | "opf"
    | "spatiotemporal"
    | "anomaly";
  showExplanations?: boolean;
}

export const ModelEquations: React.FC<ModelEquationsProps> = ({
  modelType,
  showExplanations = true,
}) => {
  const getEquationsForModel = (model: string) => {
    switch (model) {
      case "forecasting":
        return [
          {
            name: "State Space Evolution",
            latex: MathExpressions.bayesianStructuralTimeSeries.stateEvolution,
            explanation:
              "Full state space model with trend, seasonal, and regression components",
          },
          {
            name: "Observation Model",
            latex: MathExpressions.bayesianStructuralTimeSeries.observation,
            explanation:
              "Complete observation equation including all structural components",
          },
          {
            name: "MCMC Posterior",
            latex: MathExpressions.bayesianStructuralTimeSeries.mcmc,
            explanation:
              "Bayesian inference using Markov Chain Monte Carlo sampling",
          },
        ];
      case "state_estimation":
        return [
          {
            name: "Nonlinear Process Model",
            latex: MathExpressions.extendedKalmanFilter.processModel,
            explanation: "Nonlinear state evolution with process noise",
          },
          {
            name: "Nonlinear Measurement Model",
            latex: MathExpressions.extendedKalmanFilter.measurementModel,
            explanation:
              "Nonlinear measurement function with observation noise",
          },
          {
            name: "Jacobian Matrices",
            latex: MathExpressions.extendedKalmanFilter.jacobians,
            explanation: "Linearization matrices for EKF implementation",
          },
          {
            name: "State Update",
            latex: MathExpressions.extendedKalmanFilter.update,
            explanation:
              "Nonlinear state correction using measurement residual",
          },
        ];
      case "congestion":
        return [
          {
            name: "Full Posterior Distribution",
            latex: MathExpressions.mcmcLogistic.fullPosterior,
            explanation:
              "Complete Bayesian posterior for logistic regression parameters",
          },
          {
            name: "Logistic Likelihood",
            latex: MathExpressions.mcmcLogistic.logisticLikelihood,
            explanation:
              "Bernoulli likelihood function for binary classification",
          },
          {
            name: "Metropolis-Hastings Update",
            latex: MathExpressions.mcmcLogistic.metropolisStep,
            explanation: "MCMC sampling with gradient-based proposals",
          },
        ];
      case "opf":
        return [
          {
            name: "Full OPF Problem",
            latex: MathExpressions.interiorPoint.fullProblem,
            explanation:
              "Complete Optimal Power Flow with all constraints and objectives",
          },
          {
            name: "Barrier Function Formulation",
            latex: MathExpressions.interiorPoint.barrierForm,
            explanation:
              "Logarithmic barrier method for inequality constraints",
          },
          {
            name: "Newton Step",
            latex: MathExpressions.interiorPoint.newtonStep,
            explanation: "Iterative optimization using Newton's method",
          },
        ];
      case "spatiotemporal":
        return [
          {
            name: "Constrained Gaussian Process",
            latex: MathExpressions.gaussianProcessPDE.constrainedGP,
            explanation: "Gaussian Process with physical PDE constraints",
          },
          {
            name: "Physics-Informed Kernel",
            latex: MathExpressions.gaussianProcessPDE.pdeKernel,
            explanation: "Kernel function incorporating physical constraints",
          },
          {
            name: "Green's Function Kernel",
            latex: MathExpressions.gaussianProcessPDE.greenKernel,
            explanation:
              "Fundamental solution-based kernel for PDE constraints",
          },
        ];
      case "anomaly":
        return [
          {
            name: "Full HMM Likelihood",
            latex: MathExpressions.hiddenMarkovChangePoint.fullLikelihood,
            explanation:
              "Complete likelihood function with change point segmentation",
          },
          {
            name: "Regime-Based Model",
            latex: MathExpressions.hiddenMarkovChangePoint.regimeModel,
            explanation: "Hidden Markov Model with regime-specific parameters",
          },
          {
            name: "Change Point Posterior",
            latex: MathExpressions.hiddenMarkovChangePoint.changePointPosterior,
            explanation: "Bayesian inference for change point locations",
          },
        ];
      default:
        return [];
    }
  };

  const equations = getEquationsForModel(modelType);

  return (
    <div style={{ margin: "1em 0" }}>
      {equations.map((eq, idx) => (
        <div key={idx} style={{ marginBottom: "1.5em" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              marginBottom: "0.5em",
              fontWeight: "bold",
              color: "#1976d2",
            }}
          >
            {eq.name}
          </div>
          <MathEquation latex={eq.latex} />
          {showExplanations && (
            <div
              style={{
                marginTop: "0.5em",
                fontSize: "0.9em",
                color: "#666",
                fontStyle: "italic",
              }}
            >
              {eq.explanation}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};
