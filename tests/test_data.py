import numpy as np
from ai_grid_demo.data.dataset import GridTimeSeriesDataset


def test_dataset_loading():
    # Create dummy data
    data = {
        'node_features': np.random.rand(100, 5, 5),
        'node_targets': np.random.rand(100, 5, 2),
        'edge_targets': np.random.rand(100, 10, 2),
        'congestion_flags': np.random.rand(100, 10) > 0.5,
        'opf_targets': np.random.rand(100, 3),
        'edge_index': np.array([[0, 1], [1, 2]]),
        'edge_attr': np.random.rand(3, 10)
    }
    np.savez('test_data.npz', **data)

    dataset = GridTimeSeriesDataset('test_data.npz', 'state_estimation')
    assert len(dataset) > 0
    x, y, edge_index = dataset[0]
    assert x.shape == (5, 5)
    print("Dataset test passed")


if __name__ == "__main__":
    test_dataset_loading()

