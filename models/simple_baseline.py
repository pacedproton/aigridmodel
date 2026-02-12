import torch
import torch.nn as nn


class SimpleBaseline(nn.Module):
    """Very simple baseline model that should definitely converge."""

    def __init__(self, input_dim: int, n_edges: int, output_features: int = 2):
        super().__init__()
        self.n_edges = n_edges
        self.output_features = output_features
        # Just a single linear layer - this should at least learn a simple mapping
        self.linear = nn.Linear(input_dim, n_edges * output_features)

    def forward(self, x, edge_index=None):
        # x shape: (batch, seq_len, n_nodes, n_features) -> flatten to (batch, input_dim)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # Flatten all dimensions except batch
        output = self.linear(x_flat)  # (batch, n_edges * output_features)
        return output.view(batch_size, self.n_edges, self.output_features)  # (batch, n_edges, output_features)
