"""PyTorch dataset and dataloader utilities for grid time-series data."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GridTimeSeriesDataset(Dataset):
    """Sliding-window dataset over .npz grid time-series.

    Args:
        data_path: Path to .npz file produced by simulator.
        task: One of 'spatiotemporal', 'state_estimation', 'congestion'.
        history_len: Number of past time steps per sample (used for spatiotemporal).
    """

    def __init__(self, data_path: str, task: str = 'spatiotemporal', history_len: int = 8):
        data = np.load(data_path)
        self.task = task
        self.history_len = history_len

        self.node_features = data['node_features'].astype(np.float32)   # (T, N, F)
        self.node_targets = data['node_targets'].astype(np.float32)     # (T, N, Ft)
        self.edge_targets = data['edge_targets'].astype(np.float32)     # (T, E, Fe)
        self.edge_index = data['edge_index'].astype(np.int64)           # (2, E)

        self.T = self.node_features.shape[0]

    def __len__(self):
        if self.task == 'state_estimation':
            return self.T
        # For spatiotemporal / congestion we need history_len past steps + 1 target step
        return max(0, self.T - self.history_len)

    def __getitem__(self, idx):
        edge_index = torch.from_numpy(self.edge_index)

        if self.task == 'state_estimation':
            # Single-step: x=(N,F), y=(N,Ft)
            x = torch.from_numpy(self.node_features[idx])
            y = torch.from_numpy(self.node_targets[idx])
            return x, y, edge_index

        # Spatiotemporal / congestion: sliding window
        x = torch.from_numpy(self.node_features[idx: idx + self.history_len])  # (H, N, F)
        y = torch.from_numpy(self.edge_targets[idx + self.history_len])        # (E, Fe)
        return x, y, edge_index


def make_dataloaders(data_path: str, task: str, batch_size: int = 32,
                     history_len: int = 8):
    """Create train / val / test DataLoaders with a 70/15/15 split.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = GridTimeSeriesDataset(data_path, task, history_len)
    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
