#!/usr/bin/env python3
from ai_grid_demo.training.train_simple import train_simple
from ai_grid_demo.config import TrainingConfig
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


if __name__ == "__main__":
    config = TrainingConfig()
    data_path = "data/grid_data.npz"
    os.makedirs("models", exist_ok=True)
    model_path = "models/simple_mlp.pt"
    train_simple(data_path, config, model_path)
    print("Training complete.")
