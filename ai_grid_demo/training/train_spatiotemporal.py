from ..config import TrainingConfig
from ..data.dataset import make_dataloaders
from ..models.gnn_transformer import GNNTransformer
from .train_utils import train_model


def train_spatiotemporal(data_path: str, config: TrainingConfig, model_path: str):
    """Train the spatiotemporal GNN+Transformer model. Returns training history."""
    train_loader, val_loader, _ = make_dataloaders(data_path, 'spatiotemporal',
                                                  config.batch_size, config.history_len)

    # Load data to get n_edges
    import numpy as np
    data = np.load(data_path)
    n_edges = data['edge_targets'].shape[1]  # Get number of edges

    # Assuming node_dim=5, output_dim based on edge_targets (2)
    model = GNNTransformer(node_dim=5, output_dim=2, n_edges=n_edges, hidden_dim=config.hidden_dim,
                          n_gnn_layers=config.n_layers, n_trans_layers=config.n_layers,
                          n_heads=config.n_heads, dropout=config.dropout)

    history = train_model(model, train_loader, val_loader, config, model_path)
    return history

