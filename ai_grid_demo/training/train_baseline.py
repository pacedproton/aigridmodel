from ..config import TrainingConfig
from ..data.dataset import make_dataloaders
from ..models.simple_baseline import SimpleBaseline
from .train_utils import train_model


def train_baseline(data_path: str, config: TrainingConfig, model_path: str):
    """Train the simple baseline model. Returns training history."""
    train_loader, val_loader, _ = make_dataloaders(data_path, 'spatiotemporal',
                                                  config.batch_size, config.history_len)

    # Load data to get dimensions
    import numpy as np
    data = np.load(data_path)
    n_edges = data['edge_targets'].shape[1]  # Get number of edges
    n_features = data['edge_targets'].shape[2]  # Get number of features per edge

    # Calculate input dimension
    input_dim = 8 * 14 * 5  # sequence length * nodes * features

    model = SimpleBaseline(input_dim=input_dim, n_edges=n_edges, output_features=n_features)

    history = train_model(model, train_loader, val_loader, config, model_path)
    return history
