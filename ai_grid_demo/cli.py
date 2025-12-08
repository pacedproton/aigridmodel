import click
from .config import GridConfig, TrainingConfig
from .data.simulator import simulate_grid_timeseries
from .training.train_spatiotemporal import train_spatiotemporal
from .viz.plots import plot_time_series


@click.group()
def main():
    """AI Grid Demo CLI"""
    pass


@main.command()
@click.option("--output", default="data/grid_data.npz", help="Output file path")
def generate_data(output):
    """Generate synthetic grid data"""
    config = GridConfig()
    simulate_grid_timeseries(config, output)
    click.echo(f"Data generated and saved to {output}")


@main.command()
@click.option("--data-path", default="data/grid_data.npz", help="Data file path")
@click.option("--model-path", default="models/spatiotemporal.pt", help="Model save path")
def train_spatiotemporal(data_path, model_path):
    """Train spatiotemporal GNN+Transformer model"""
    config = TrainingConfig()
    train_spatiotemporal(data_path, config, model_path)
    click.echo(f"Model trained and saved to {model_path}")


@main.command()
@click.option("--model-path", default="models/spatiotemporal.pt", help="Model file path")
@click.option("--data-path", default="data/grid_data.npz", help="Data file path")
def evaluate(model_path, data_path):
    """Evaluate model and generate plots"""
    # Load model and data, compute metrics, plot
    plot_time_series(data_path)  # Placeholder
    click.echo("Evaluation complete")


if __name__ == "__main__":
    main()
