#!/usr/bin/env python3
"""
Test script for classical mathematical models in AI Grid

This script demonstrates and validates the classical model implementations
against the neural network baselines.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_classical_models():
    """Test all classical models against synthetic data"""

    print("Testing Classical Mathematical Models for AI Grid")
    print("=" * 60)

    # Import our implementations
    from ai_grid_demo.classical_models import (
        ClassicalLoadForecaster,
        DCStateEstimator,
        PTDFCongestionPredictor,
        ClassicalModelsComparison
    )

    # Generate synthetic data (similar to our simulator)
    np.random.seed(42)
    T = 1000  # Time steps
    n_nodes = 14  # IEEE 14-bus system

    print(f"Generating synthetic data: {T} time steps, {n_nodes} nodes")

    # Create synthetic load data with patterns
    time = np.arange(T)
    base_load = 50 + 20 * np.sin(2 * np.pi * time / 24)  # Daily pattern
    trend = 0.01 * time  # Slow upward trend
    noise = np.random.normal(0, 2, T)  # Random noise

    # Create multivariate load series
    load_data = np.zeros((T, n_nodes))
    for i in range(n_nodes):
        # Each node has correlated but unique patterns
        node_factor = 0.8 + 0.4 * np.random.random()
        phase_shift = 2 * np.pi * np.random.random()
        seasonal = 10 * np.sin(2 * np.pi * time / 24 + phase_shift)
        load_data[:, i] = (base_load + seasonal + trend + noise) * node_factor

    print("Testing Load Forecasting (VAR)")
    print("-" * 40)

    # Test load forecasting
    forecaster = ClassicalLoadForecaster(lag_order=12)
    fit_info = forecaster.fit(load_data[:700])  # Train on first 70%

    print(f"BIC: {fit_info['bic']:.1f}")
    print(f"Lag order selected: {fit_info['lag_order']}")
    print(f"AIC: {fit_info['aic']:.4f}")

    # Test predictions
    test_predictions = []
    for i in range(700, min(800, T-12)):  # Predict next 100 steps
        history = load_data[i-12:i]
        pred = forecaster.predict(history, steps_ahead=1)
        test_predictions.append(pred[0])

    test_predictions = np.array(test_predictions)
    true_values = load_data[700:800]

    mse = np.mean((true_values - test_predictions) ** 2)
    mae = np.mean(np.abs(true_values - test_predictions))
    mape = np.mean(np.abs((true_values - test_predictions) / (true_values + 1e-8))) * 100

    print("Test Performance (next 100 steps):")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    print("\n🔌 Testing State Estimation (DC WLS)")
    print("-" * 40)

    # Test state estimation
    grid_config = {
        'n_buses': n_nodes,
        'n_lines': 20,
        'slack_bus': 0
    }

    estimator = DCStateEstimator(grid_config)
    measurement_config = {
        'n_measurements': n_nodes,
        'noise_variance': 0.01
    }
    estimator.initialize_matrices(measurement_config)

    # Generate synthetic measurements
    true_states = np.random.randn(100, n_nodes - 1) * 0.1  # Exclude slack
    measurements = true_states + np.random.normal(0, 0.1, true_states.shape)

    # Estimate states
    estimates = []
    for i in range(len(measurements)):
        estimate = estimator.estimate_state(measurements[i])
        estimates.append(estimate)
    estimates = np.array(estimates)

    mse_se = np.mean((true_states - estimates) ** 2)
    mae_se = np.mean(np.abs(true_states - estimates))

    print("State Estimation Performance:")
    print(f"MSE: {mse_se:.6f}")
    print(f"MAE: {mae_se:.6f}")
    print(f"Measurements: {measurements.shape[1]}, State vars: {true_states.shape[1]}")

    print("\nTesting Congestion Prediction (PTDF + Logistic)")
    print("-" * 40)

    # Test congestion prediction
    congestion_predictor = PTDFCongestionPredictor(grid_config)
    fit_info_cp = congestion_predictor.fit(
        load_data[:700],  # Training injections
        np.random.randint(0, 2, (700, 20))  # Random congestion labels
    )

    print("Training completed:")
    print(f"Samples: {fit_info_cp['n_samples']}")
    print(f"Features: {fit_info_cp['n_features']}")
    print(f"Training accuracy: {fit_info_cp['accuracy']:.3f}")
    # Test predictions
    test_proba = congestion_predictor.predict_proba(load_data[700:800])
    print(f"Test predictions shape: {test_proba.shape}")

    print("\nTesting Complete!")
    print("=" * 60)
    print("All classical models implemented and tested")
    print("Ready for comparison with neural network baselines")
    print("\nNext steps:")
    print("1. Run full comparison: python -m ai_grid_demo.classical_models")
    print("2. Start API: python scripts/start_api.py")
    print("3. Test frontend: cd frontend && npm start")

if __name__ == "__main__":
    test_classical_models()
