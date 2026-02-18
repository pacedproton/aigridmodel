"""Synthetic IEEE 14-bus power grid time-series generator."""

import numpy as np
from ..config import GridConfig
from .grid_generator import build_ieee_14_grid


def simulate_grid_timeseries(config: GridConfig, output_path: str) -> None:
    """Generate synthetic IEEE 14-bus power grid time series and save to .npz.

    Produces keys:
        node_features  (T, 14, 5)  — P_load, Q_load, P_gen, renewable_injection, voltage_setpoint
        node_targets   (T, 14, 2)  — voltage_magnitude, voltage_angle
        edge_targets   (T, 20, 2)  — active_power_flow, reactive_power_flow
        congestion_flags (T, 20)   — binary congestion indicator per line
        edge_index     (2, 20)     — graph connectivity (from, to)
        edge_attr      (20, 4)     — static line attributes (r, x, b, rating)
    """
    grid = build_ieee_14_grid()
    n_buses = grid['n_buses']       # 14
    n_lines = grid['n_lines']       # 20
    T = config.n_steps
    rng = np.random.default_rng(42)

    # --- time axis ---
    t = np.arange(T)
    dt_hours = config.dt_minutes / 60.0

    # --- base load profile (daily sinusoid + trend) ---
    if config.load_pattern_type == 'daily_sinusoid':
        # 24-hour cycle
        period = 24.0 / dt_hours  # steps per day
        base_pattern = 0.7 + 0.3 * np.sin(2 * np.pi * t / period - np.pi / 2)
    else:
        base_pattern = np.ones(T)

    # Per-bus load scaling (different buses have different base loads)
    bus_load_scale = rng.uniform(0.5, 2.0, size=n_buses)

    # --- node features (T, 14, 5) ---
    node_features = np.zeros((T, n_buses, 5), dtype=np.float32)

    for b in range(n_buses):
        # P_load
        p_load = base_pattern * bus_load_scale[b] + config.noise_level * rng.normal(size=T)
        node_features[:, b, 0] = p_load

        # Q_load (reactive, ~30% of active)
        q_load = 0.3 * p_load + 0.05 * config.noise_level * rng.normal(size=T)
        node_features[:, b, 1] = q_load

        # P_gen (only generator buses produce)
        gen_buses = [g['bus'] for g in grid['generators']]
        if b in gen_buses:
            node_features[:, b, 2] = np.abs(p_load) * 1.1 + config.noise_level * 0.1 * rng.normal(size=T)

        # Renewable injection
        solar_pattern = np.clip(np.sin(2 * np.pi * t / (24.0 / dt_hours) - np.pi / 3), 0, 1)
        node_features[:, b, 3] = config.renewable_share * solar_pattern * bus_load_scale[b] + \
            0.05 * config.noise_level * rng.normal(size=T)

        # Voltage setpoint (near 1.0 pu)
        node_features[:, b, 4] = 1.0 + 0.02 * rng.normal(size=T)

    # --- node targets (T, 14, 2) ---
    node_targets = np.zeros((T, n_buses, 2), dtype=np.float32)
    for b in range(n_buses):
        # Voltage magnitude ~ 1.0 ± small perturbation driven by load
        node_targets[:, b, 0] = 1.0 - 0.02 * node_features[:, b, 0] / bus_load_scale.max() + \
            0.005 * config.noise_level * rng.normal(size=T)
        # Voltage angle (radians, small)
        node_targets[:, b, 1] = -0.05 * node_features[:, b, 0] / bus_load_scale.max() + \
            0.01 * config.noise_level * rng.normal(size=T)

    # Slack bus angle = 0
    node_targets[:, grid['slack_bus'], 1] = 0.0

    # --- edge index (2, 20) ---
    from_nodes = np.array([l[0] for l in grid['lines']], dtype=np.int64)
    to_nodes = np.array([l[1] for l in grid['lines']], dtype=np.int64)
    edge_index = np.stack([from_nodes, to_nodes], axis=0)

    # --- edge attributes (20, 4) : r, x, b, rating ---
    edge_attr = np.zeros((n_lines, 4), dtype=np.float32)
    edge_attr[:, 0] = rng.uniform(0.01, 0.1, size=n_lines)   # resistance
    edge_attr[:, 1] = rng.uniform(0.05, 0.3, size=n_lines)   # reactance
    edge_attr[:, 2] = rng.uniform(0.01, 0.05, size=n_lines)  # susceptance
    edge_attr[:, 3] = rng.uniform(50, 150, size=n_lines)      # thermal rating

    # --- edge targets (T, 20, 2) ---
    edge_targets = np.zeros((T, n_lines, 2), dtype=np.float32)
    for e, (i, j) in enumerate(grid['lines']):
        # Active power flow ~ proportional to angle difference / reactance
        angle_diff = node_targets[:, i, 1] - node_targets[:, j, 1]
        edge_targets[:, e, 0] = angle_diff / edge_attr[e, 1] + \
            0.02 * config.noise_level * rng.normal(size=T)
        # Reactive power flow
        v_diff = node_targets[:, i, 0] - node_targets[:, j, 0]
        edge_targets[:, e, 1] = v_diff / edge_attr[e, 1] + \
            0.01 * config.noise_level * rng.normal(size=T)

    # --- congestion flags (T, 20) ---
    line_ratings = edge_attr[:, 3]
    flow_magnitude = np.abs(edge_targets[:, :, 0])
    # Congestion when flow exceeds 80% of rating
    congestion_flags = (flow_magnitude > 0.8 * line_ratings[np.newaxis, :]).astype(np.float32)

    np.savez(
        output_path,
        node_features=node_features,
        node_targets=node_targets,
        edge_targets=edge_targets,
        congestion_flags=congestion_flags,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    print(f"Saved grid data to {output_path}: T={T}, nodes={n_buses}, edges={n_lines}")
