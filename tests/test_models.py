import torch
from ai_grid_demo.models.gnn_transformer import GNNTransformer


def test_gnn_transformer():
    model = GNNTransformer(node_dim=5, output_dim=2)
    x_seq = torch.randn(1, 8, 5, 5)  # batch=1, seq=8, nodes=5, dim=5
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    output = model(x_seq, edge_index)
    assert output.shape == (1, 2)
    print("GNNTransformer test passed")


if __name__ == "__main__":
    test_gnn_transformer()

