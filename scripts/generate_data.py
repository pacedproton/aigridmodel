#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_grid_demo.config import GridConfig
from ai_grid_demo.data.simulator import simulate_grid_timeseries

if __name__ == "__main__":
    config = GridConfig()
    os.makedirs("data", exist_ok=True)
    simulate_grid_timeseries(config, "data/grid_data.npz")
    print("Data generated.")
