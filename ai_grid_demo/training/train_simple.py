from ..config import TrainingConfig
from ..data.dataset import make_dataloaders
from ..models.simple_mlp import SimpleMLP
from .train_utils import train_model


def train_simple(data_path: str, config: TrainingConfig, model_path: str):
    """Train the simple MLP model."""
    train_loader, val_loader, _ = make_dataloaders(data_path, 'spatiotemporal',
                                                  config.batch_size, config.history_len)

    # Calculate input dim: history_len * n_nodes * node_dim
    # From data: n_nodes=14, node_dim=5, history_len=8 -> 14*5*8=560
    input_dim = 8 * 14 * 5
    output_dim = 2  # edge_targets: 2 dims

    model = SimpleMLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=config.hidden_dim)

    train_model(model, train_loader, val_loader, config, model_path)

