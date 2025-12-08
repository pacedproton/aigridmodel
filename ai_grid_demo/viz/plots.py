import numpy as np
import matplotlib.pyplot as plt


def plot_time_series(data_path: str):
    """Plot sample time series from data."""
    data = np.load(data_path)
    node_features = data['node_features']
    node_targets = data['node_targets']

    # Plot first node's load and voltage over time
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(node_features[:1000, 0, 0], label='P_load')
    plt.title('Load Time Series')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(node_targets[:1000, 0, 0], label='Voltage')
    plt.title('Voltage Time Series')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/sample_timeseries.png')
    plt.show()

