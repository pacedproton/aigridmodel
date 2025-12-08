from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GridConfig:
    grid_type: str = "case14"  # IEEE case
    n_steps: int = 2000  # More data for better training
    dt_minutes: int = 30
    noise_level: float = 0.2  # Higher noise for more challenging learning
    load_pattern_type: str = "daily_sinusoid"
    renewable_share: float = 0.3  # More renewables for complexity
    opf_fraction: float = 0.1  # Fraction of steps to run OPF


@dataclass
class TrainingConfig:
    batch_size: int = 32  # Larger batch for stability
    epochs: int = 100  # Many epochs to show full convergence
    lr: float = 1e-4  # Very low learning rate for stable convergence
    history_len: int = 8
    hidden_dim: int = 64  # Simpler model first
    n_layers: int = 2  # Simpler architecture
    n_heads: int = 4
    dropout: float = 0.1  # Lower dropout
    patience: int = 50  # Much more patience
    random_seed: int = 42


@dataclass
class ModelConfig:
    model_type: str  # "temporal_cnn", "gnn_state", etc.
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    task: str = "regression"  # or "classification"


@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(model_type="spatiotemporal"))
