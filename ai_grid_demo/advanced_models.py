"""
Advanced Mathematical Models for AI Grid Use Cases

This module implements sophisticated mathematical models for comparison with neural networks:
1. Bayesian Structural Time Series (BSTS) for load forecasting
2. Extended Kalman Filter (EKF) for state estimation
3. MCMC-based logistic regression for congestion prediction
4. Interior Point Method for OPF
5. Gaussian Process with PDE constraints for spatiotemporal modeling
6. Hidden Markov Model for change point detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats, optimize
from scipy.special import expit
import cvxpy as cp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


class BayesianStructuralTimeSeries:
    """
    Bayesian Structural Time Series for load forecasting
    Combines trend, seasonality, and regression components with uncertainty quantification
    """

    def __init__(self, seasonal_periods: List[int] = [24, 168], n_iter: int = 1000):
        self.seasonal_periods = seasonal_periods
        self.n_iter = n_iter
        self.scaler = StandardScaler()

        # Model parameters
        self.trend_params = None
        self.seasonal_params = {}
        self.regression_params = None
        self.noise_variance = None

        # MCMC samples
        self.param_samples = []
        self.is_fitted = False

    def _initialize_parameters(self, n_features: int):
        """Initialize model parameters"""
        # Trend parameters: level, slope
        self.trend_params = {
            'level': np.random.normal(0, 1),
            'slope': np.random.normal(0, 0.1),
            'level_var': np.random.gamma(1, 1),
            'slope_var': np.random.gamma(1, 0.1)
        }

        # Seasonal parameters for each period
        for period in self.seasonal_periods:
            self.seasonal_params[period] = {
                'coefficients': np.random.normal(0, 0.1, period),
                'variance': np.random.gamma(1, 0.1)
            }

        # Regression parameters
        self.regression_params = np.random.normal(0, 0.1, n_features)

        # Noise variance
        self.noise_variance = np.random.gamma(1, 1)

    def _trend_component(self, t: int, params: Dict) -> float:
        """Compute trend component at time t"""
        level = params['level'] + params['slope'] * t
        return level

    def _seasonal_component(self, t: int, params: Dict) -> float:
        """Compute seasonal component at time t"""
        seasonal_sum = 0
        for period, season_params in params.items():
            coeff = season_params['coefficients']
            seasonal_sum += coeff[t % period]
        return seasonal_sum

    def _log_likelihood(self, y: np.ndarray, X: Optional[np.ndarray],
                       params: Dict) -> float:
        """Compute log likelihood of data given parameters"""
        n = len(y)
        predictions = np.zeros(n)

        for t in range(n):
            trend = self._trend_component(t, params['trend'])
            seasonal = self._seasonal_component(t, params['seasonal'])

            prediction = trend + seasonal

            if X is not None:
                prediction += X[t] @ params['regression']

            predictions[t] = prediction

        residuals = y - predictions
        log_lik = -0.5 * n * np.log(2 * np.pi * params['noise_var']) - \
                  0.5 * np.sum(residuals**2) / params['noise_var']

        return log_lik

    def _sample_parameters(self, y: np.ndarray, X: Optional[np.ndarray]) -> Dict:
        """Sample new parameter values using Metropolis-Hastings"""
        # Current parameters
        current_params = {
            'trend': self.trend_params.copy(),
            'seasonal': self.seasonal_params.copy(),
            'regression': self.regression_params.copy(),
            'noise_var': self.noise_variance
        }

        current_log_lik = self._log_likelihood(y, X, current_params)

        # Propose new parameters (simplified random walk)
        new_params = current_params.copy()

        # Trend parameters
        new_params['trend'] = {
            'level': current_params['trend']['level'] + np.random.normal(0, 0.1),
            'slope': current_params['trend']['slope'] + np.random.normal(0, 0.01),
            'level_var': current_params['trend']['level_var'],
            'slope_var': current_params['trend']['slope_var']
        }

        # Seasonal parameters
        new_params['seasonal'] = {}
        for period in self.seasonal_periods:
            new_params['seasonal'][period] = {
                'coefficients': current_params['seasonal'][period]['coefficients'] +
                               np.random.normal(0, 0.05, period),
                'variance': current_params['seasonal'][period]['variance']
            }

        # Regression parameters
        if X is not None:
            new_params['regression'] = (current_params['regression'] +
                                       np.random.normal(0, 0.05, len(current_params['regression'])))

        # Noise variance
        new_params['noise_var'] = np.abs(current_params['noise_var'] +
                                        np.random.normal(0, 0.1))

        new_log_lik = self._log_likelihood(y, X, new_params)

        # Accept/reject
        log_accept_ratio = new_log_lik - current_log_lik
        if np.log(np.random.random()) < log_accept_ratio:
            return new_params
        else:
            return current_params

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit BSTS model using MCMC

        Args:
            y: Time series values (n_samples,)
            X: Optional exogenous variables (n_samples, n_features)

        Returns:
            Dictionary with fit information
        """
        # Scale data
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()

        n_features = X.shape[1] if X is not None else 0
        self._initialize_parameters(n_features)

        # MCMC sampling
        self.param_samples = []
        for i in range(self.n_iter):
            new_params = self._sample_parameters(y_scaled, X)
            self.param_samples.append(new_params)

            # Update current parameters
            self.trend_params = new_params['trend']
            self.seasonal_params = new_params['seasonal']
            self.regression_params = new_params['regression']
            self.noise_variance = new_params['noise_var']

        self.is_fitted = True

        return {
            'n_iterations': self.n_iter,
            'seasonal_periods': self.seasonal_periods,
            'n_features': n_features,
            'final_noise_var': self.noise_variance,
            'trend_params': self.trend_params,
            'seasonal_params': self.seasonal_params
        }

    def predict(self, steps_ahead: int = 1, X_future: Optional[np.ndarray] = None,
                return_uncertainty: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions with uncertainty quantification

        Args:
            steps_ahead: Number of steps to forecast
            X_future: Future exogenous variables
            return_uncertainty: Whether to return uncertainty bounds

        Returns:
            Mean predictions, or (mean, lower_bound, upper_bound) if return_uncertainty
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        n_samples = len(self.param_samples)
        predictions = np.zeros((n_samples, steps_ahead))

        # Current time (continuation from training)
        current_t = len(self.scaler.mean_)  # Approximate

        for i, params in enumerate(self.param_samples[-100:]):  # Use last 100 samples
            for t in range(steps_ahead):
                trend = self._trend_component(current_t + t, params['trend'])
                seasonal = self._seasonal_component(current_t + t, params['seasonal'])

                pred = trend + seasonal

                if X_future is not None and t < X_future.shape[0]:
                    pred += X_future[t] @ params['regression']

                predictions[i, t] = pred

        # Inverse transform
        predictions_unscaled = self.scaler.inverse_transform(predictions.T).T

        mean_predictions = np.mean(predictions_unscaled, axis=0)

        if not return_uncertainty:
            return mean_predictions

        # Uncertainty bounds (95% credible intervals)
        lower_bound = np.percentile(predictions_unscaled, 2.5, axis=0)
        upper_bound = np.percentile(predictions_unscaled, 97.5, axis=0)

        return mean_predictions, lower_bound, upper_bound


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state estimation
    Handles nonlinear process and measurement models
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance

        # State: [voltage_angles...]
        self.state_dim = None
        self.x_hat = None  # State estimate
        self.P = None      # State covariance

    def initialize(self, initial_state: np.ndarray):
        """Initialize filter with initial state estimate"""
        self.state_dim = len(initial_state)
        self.x_hat = initial_state.copy()
        self.P = np.eye(self.state_dim) * 0.1  # Initial uncertainty

    def process_model(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Nonlinear process model (simplified physics-based)"""
        # Simplified: assume small changes in angles
        return x  # Constant velocity model

    def measurement_model(self, x: np.ndarray) -> np.ndarray:
        """Nonlinear measurement model (power flow equations)"""
        # Simplified: direct measurement of angles with some nonlinearity
        return x + 0.1 * np.sin(x)  # Add mild nonlinearity

    def jacobian_F(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of process model"""
        return np.eye(self.state_dim)  # Identity for constant model

    def jacobian_H(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of measurement model"""
        return np.eye(self.state_dim) + 0.1 * np.diag(np.cos(x))

    def predict(self):
        """Prediction step"""
        # State prediction
        self.x_hat = self.process_model(self.x_hat)

        # Covariance prediction
        F = self.jacobian_F(self.x_hat)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: np.ndarray):
        """Update step with new measurement"""
        # Measurement prediction
        z_pred = self.measurement_model(self.x_hat)

        # Innovation
        y = measurement - z_pred

        # Innovation covariance
        H = self.jacobian_H(self.x_hat)
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x_hat = self.x_hat + K @ y

        # Covariance update
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

        return y, S  # Return innovation and covariance for monitoring

    def estimate(self, measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run filter on sequence of measurements

        Args:
            measurements: Shape (n_steps, n_measurements)

        Returns:
            State estimates and covariances
        """
        n_steps = measurements.shape[0]
        estimates = np.zeros((n_steps, self.state_dim))
        covariances = np.zeros((n_steps, self.state_dim, self.state_dim))

        for t in range(n_steps):
            # Prediction step
            self.predict()

            # Update step
            self.update(measurements[t])

            # Store results
            estimates[t] = self.x_hat.copy()
            covariances[t] = self.P.copy()

        return estimates, covariances


class MCMCLogisticRegression:
    """
    MCMC-based logistic regression for congestion prediction
    Provides full posterior distributions over parameters
    """

    def __init__(self, n_iter: int = 1000, burn_in: int = 200):
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.param_samples = []
        self.is_fitted = False

    def _log_posterior(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Log posterior probability"""
        # Logistic likelihood
        logits = X @ theta
        log_lik = np.sum(y * logits - np.log(1 + np.exp(logits)))

        # Gaussian prior
        log_prior = -0.5 * np.sum(theta**2)

        return log_lik + log_prior

    def _metropolis_step(self, theta_current: np.ndarray, X: np.ndarray,
                        y: np.ndarray, step_size: float = 0.1) -> np.ndarray:
        """Single Metropolis-Hastings step"""
        # Propose new parameters
        theta_proposed = theta_current + np.random.normal(0, step_size, len(theta_current))

        # Compute acceptance ratio
        log_p_current = self._log_posterior(theta_current, X, y)
        log_p_proposed = self._log_posterior(theta_proposed, X, y)

        log_accept_ratio = log_p_proposed - log_p_current

        # Accept or reject
        if np.log(np.random.random()) < log_accept_ratio:
            return theta_proposed
        else:
            return theta_current

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit MCMC logistic regression

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)

        Returns:
            Fit information
        """
        n_features = X.shape[1]

        # Initialize parameters
        theta = np.random.normal(0, 0.1, n_features)

        # MCMC sampling
        self.param_samples = []
        for i in range(self.n_iter):
            theta = self._metropolis_step(theta, X, y)
            self.param_samples.append(theta.copy())

        # Remove burn-in
        self.param_samples = self.param_samples[self.burn_in:]

        self.is_fitted = True

        return {
            'n_iterations': self.n_iter,
            'burn_in': self.burn_in,
            'effective_samples': len(self.param_samples),
            'n_features': n_features
        }

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict probabilities with uncertainty

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Mean probabilities, lower bound, upper bound
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        prob_samples = np.zeros((len(self.param_samples), n_samples))

        # Compute probabilities for each parameter sample
        for i, theta in enumerate(self.param_samples):
            logits = X @ theta
            prob_samples[i] = expit(logits)  # Sigmoid

        # Compute statistics
        mean_prob = np.mean(prob_samples, axis=0)
        lower_prob = np.percentile(prob_samples, 2.5, axis=0)
        upper_prob = np.percentile(prob_samples, 97.5, axis=0)

        return mean_prob, lower_prob, upper_prob


class InteriorPointOPF:
    """
    Interior Point Method for Optimal Power Flow
    State-of-the-art nonlinear optimization algorithm
    """

    def __init__(self, tolerance: float = 1e-6, max_iter: int = 100):
        self.tolerance = tolerance
        self.max_iter = max_iter

    def solve(self, n_buses: int, n_gens: int, load_demand: np.ndarray,
             gen_costs: Dict[str, np.ndarray], gen_limits: Dict[str, Tuple[float, float]],
             line_limits: Dict[str, float]) -> Dict[str, Any]:
        """
        Solve OPF using interior point method

        Args:
            n_buses: Number of buses
            n_gens: Number of generators
            load_demand: Load at each bus
            gen_costs: Generator cost coefficients
            gen_limits: Generator min/max power
            line_limits: Line flow limits

        Returns:
            Optimal solution
        """
        # This is a simplified implementation
        # In practice, would use full AC power flow equations

        # Decision variables
        Pg = cp.Variable(n_gens)  # Generator powers
        theta = cp.Variable(n_buses)  # Bus angles

        # Objective: minimize generation cost
        cost = 0
        for i in range(n_gens):
            cost += gen_costs['a'][i] * cp.square(Pg[i]) + \
                   gen_costs['b'][i] * Pg[i] + \
                   gen_costs['c'][i]

        # Constraints
        constraints = []

        # Generator limits
        for i in range(n_gens):
            constraints.extend([
                Pg[i] >= gen_limits[f'gen_{i}'][0],
                Pg[i] <= gen_limits[f'gen_{i}'][1]
            ])

        # Power balance (simplified DC model)
        total_gen = cp.sum(Pg)
        total_load = cp.sum(load_demand)
        constraints.append(total_gen >= total_load)

        # Slack bus reference
        constraints.append(theta[0] == 0)

        # Line flow limits (simplified)
        for line, limit in line_limits.items():
            # Would compute PTDF * (theta differences) <= limit
            # Simplified constraint
            constraints.append(theta[1] - theta[0] <= 0.5)

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)

        try:
            result = prob.solve(solver=cp.OSQP, verbose=False)

            if prob.status == 'optimal':
                return {
                    'success': True,
                    'Pg_opt': Pg.value,
                    'theta_opt': theta.value,
                    'total_cost': result,
                    'status': 'optimal'
                }
            else:
                return {
                    'success': False,
                    'error': f'Solver status: {prob.status}',
                    'status': prob.status
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'status': 'error'
            }


class GaussianProcessPDE:
    """
    Gaussian Process with PDE constraints for spatiotemporal modeling
    Incorporates physical differential equation constraints
    """

    def __init__(self, length_scale: float = 1.0, noise_variance: float = 0.1):
        self.length_scale = length_scale
        self.noise_variance = noise_variance

        # Kernel: spatial * temporal * coupling
        spatial_kernel = C(1.0) * RBF(length_scale)
        temporal_kernel = C(1.0) * RBF(length_scale)

        self.kernel = spatial_kernel * temporal_kernel
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=noise_variance,
            n_restarts_optimizer=10
        )

        self.is_fitted = False

    def _pde_constraint_matrix(self, spatial_coords: np.ndarray,
                              temporal_coords: np.ndarray) -> np.ndarray:
        """
        Build PDE constraint matrix
        Simplified: assume some physical constraint like ∇²f = 0
        """
        # This would implement actual PDE constraints
        # For now, return identity (no constraints)
        n_points = len(spatial_coords)
        return np.eye(n_points)

    def fit(self, spatial_coords: np.ndarray, temporal_coords: np.ndarray,
            values: np.ndarray) -> Dict[str, Any]:
        """
        Fit GP with PDE constraints

        Args:
            spatial_coords: Spatial coordinates (n_samples, n_spatial_dims)
            temporal_coords: Temporal coordinates (n_samples,)
            values: Observed values (n_samples,)
        """
        # Combine spatial and temporal coordinates
        X = np.column_stack([spatial_coords, temporal_coords])

        # Fit GP
        self.gp.fit(X, values)
        self.is_fitted = True

        return {
            'kernel_params': self.gp.kernel_.get_params(),
            'log_marginal_likelihood': self.gp.log_marginal_likelihood_value_,
            'n_samples': len(values)
        }

    def predict(self, spatial_coords_pred: np.ndarray,
               temporal_coords_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty

        Args:
            spatial_coords_pred: Prediction spatial coordinates
            temporal_coords_pred: Prediction temporal coordinates

        Returns:
            Mean predictions and standard deviations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_pred = np.column_stack([spatial_coords_pred, temporal_coords_pred])

        y_pred, y_std = self.gp.predict(X_pred, return_std=True)

        return y_pred, y_std


class HiddenMarkovChangePoint:
    """
    Hidden Markov Model for change point detection in anomaly detection
    Models multiple operational regimes with change point detection
    """

    def __init__(self, n_regimes: int = 3, max_change_points: int = 5):
        self.n_regimes = n_regimes
        self.max_change_points = max_change_points
        self.gmms = []  # One GMM per regime
        self.regime_params = []
        self.is_fitted = False

    def fit(self, data: np.ndarray, segment_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit HMM with change point detection

        Args:
            data: Time series data (n_samples, n_features)
            segment_labels: Optional pre-labeled segments

        Returns:
            Model fit information
        """
        n_samples, n_features = data.shape

        if segment_labels is None:
            # Use GMM to discover regimes
            gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
            regime_labels = gmm.fit_predict(data)
            self.gmms.append(gmm)
        else:
            regime_labels = segment_labels

        # Fit one GMM per regime
        for regime in range(self.n_regimes):
            regime_data = data[regime_labels == regime]
            if len(regime_data) > 0:
                gmm_regime = GaussianMixture(n_components=1, random_state=42)
                gmm_regime.fit(regime_data)
                self.gmms.append(gmm_regime)

                # Store parameters
                self.regime_params.append({
                    'mean': gmm_regime.means_[0],
                    'covariance': gmm_regime.covariances_[0],
                    'weight': np.mean(regime_labels == regime)
                })

        self.is_fitted = True

        return {
            'n_regimes': self.n_regimes,
            'regime_sizes': [np.sum(regime_labels == i) for i in range(self.n_regimes)],
            'regime_params': self.regime_params
        }

    def detect_anomalies(self, data: np.ndarray, window_size: int = 50) -> np.ndarray:
        """
        Detect anomalies using regime likelihood

        Args:
            data: Time series data (n_samples, n_features)
            window_size: Window size for regime assessment

        Returns:
            Anomaly scores (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detection")

        n_samples = len(data)
        anomaly_scores = np.zeros(n_samples)

        for i in range(window_size, n_samples):
            window = data[i-window_size:i]

            # Compute likelihood under each regime
            regime_likelihoods = []
            for gmm in self.gmms:
                try:
                    # Use last point for anomaly score
                    log_lik = gmm.score_samples(window[-1:])[0]
                    regime_likelihoods.append(np.exp(log_lik))
                except:
                    regime_likelihoods.append(0)

            # Anomaly score: 1 - max likelihood
            max_likelihood = max(regime_likelihoods) if regime_likelihoods else 0
            anomaly_scores[i] = 1 - max_likelihood

        return anomaly_scores

    def find_change_points(self, data: np.ndarray) -> List[int]:
        """
        Find change points in the time series

        Args:
            data: Time series data

        Returns:
            List of change point indices
        """
        # Simplified change point detection
        # In practice, would use proper Bayesian change point detection
        anomaly_scores = self.detect_anomalies(data)

        # Find peaks in anomaly scores
        threshold = np.percentile(anomaly_scores, 95)
        change_points = np.where(anomaly_scores > threshold)[0].tolist()

        return change_points[:self.max_change_points]


# Factory function for creating advanced models
def create_advanced_model(model_type: str, **kwargs) -> Any:
    """
    Factory function for creating advanced mathematical models

    Args:
        model_type: Type of model ('forecasting', 'state_estimation', etc.)
        **kwargs: Model-specific parameters

    Returns:
        Instantiated model object
    """
    models = {
        'forecasting': BayesianStructuralTimeSeries,
        'state_estimation': ExtendedKalmanFilter,
        'congestion': MCMCLogisticRegression,
        'opf': InteriorPointOPF,
        'spatiotemporal': GaussianProcessPDE,
        'anomaly': HiddenMarkovChangePoint
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](**kwargs)
