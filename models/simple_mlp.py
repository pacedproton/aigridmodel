import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple MLP for spatiotemporal modeling (flattened input)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index=None):
        # x: (batch, seq_len, n_nodes, node_dim) -> flatten to (batch, seq_len * n_nodes * node_dim)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.net(x_flat)
