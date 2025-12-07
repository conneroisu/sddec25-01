# Train Ellipse

Train ellipse regression model for pupil detection.

## Usage

```bash
# Install dependencies
uv pip install -e .

# Run training
train-ellipse --epochs 15 --batch-size 4 --lr 0.001
```

## Arguments

- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.001)
- `--output-dir`: Output directory for checkpoints (default: ./checkpoints)
- `--device`: Device to use (auto/cuda/cpu, default: auto)
- `--no-mlflow`: Disable MLflow logging
- `--channel-size`: Base channel size for the model (default: 32)
- `--dropout-prob`: Dropout probability (default: 0.2)
- `--num-workers`: Number of data loader workers (default: 4)

## Environment Variables (for MLflow)

- `MLFLOW_TRACKING_URI`: MLflow tracking server URI
- `MLFLOW_EXPERIMENT_ID`: MLflow experiment ID

## Example with MLflow

```bash
export MLFLOW_TRACKING_URI="https://your-mlflow-server"
export MLFLOW_EXPERIMENT_ID="123456"
train-ellipse --epochs 20 --batch-size 8
```
