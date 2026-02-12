import torch
import torch.nn as nn


class GraphEncoder(nn.Module):
    """Simplified graph encoder using linear layers (no torch_geometric)."""

    def __init__(self, node_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(node_dim, hidden_dim)] +
                                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

    def forward(self, x, edge_index):
        # x: (1, n_nodes, node_dim) - single sample
        x = x.squeeze(0)  # Remove batch dim: (n_nodes, node_dim)
        for layer in self.layers:
            x = layer(x).relu()
        # Simple mean pool over nodes
        return x.mean(dim=0, keepdim=True)  # (1, hidden_dim)


class TemporalEncoder(nn.Module):
    """Transformer encoder for temporal sequences."""

    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, batch_first=False),
            num_layers=n_layers
        )

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return self.encoder(x)


class PredictionHead(nn.Module):
    """Final prediction head."""

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, x):
        return self.linear(x)


class GNNTransformer(nn.Module):
    """GNN + Transformer for spatiotemporal modeling."""

    def __init__(self, node_dim: int, output_dim: int, n_edges: int, hidden_dim: int = 64, n_gnn_layers: int = 2,
                 n_trans_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_edges = n_edges
        self.graph_encoder = GraphEncoder(node_dim, hidden_dim, n_gnn_layers)
        self.temporal_encoder = TemporalEncoder(hidden_dim, n_trans_layers, n_heads, dropout)
        self.head = PredictionHead(hidden_dim, output_dim * n_edges)

    def forward(self, x_seq, edge_index):
        # x_seq: (batch, seq_len, n_nodes, node_dim)
        batch_size, seq_len, n_nodes, node_dim = x_seq.shape

        # Process each sample in the batch
        outputs = []
        for b in range(batch_size):
            x_seq_b = x_seq[b]  # (seq_len, n_nodes, node_dim)

            graph_embs = []
            for t in range(seq_len):
                x_t = x_seq_b[t, :, :]  # (n_nodes, node_dim)
                emb = self.graph_encoder(x_t.unsqueeze(0), edge_index)  # (1, hidden_dim)
                graph_embs.append(emb.squeeze(0))  # (hidden_dim,)

            temporal_input = torch.stack(graph_embs, dim=0).unsqueeze(1)  # (seq_len, 1, hidden_dim)
            encoded = self.temporal_encoder(temporal_input)
            output_b = self.head(encoded[-1, 0])  # Last token, (output_dim * n_edges,)
            output_b = output_b.view(self.n_edges, -1)  # (n_edges, output_dim)
            outputs.append(output_b)

        return torch.stack(outputs, dim=0)  # (batch_size, n_edges, output_dim)
