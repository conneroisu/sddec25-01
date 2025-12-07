# Train Ellipse (Inference-Aligned)

Train ellipse regression model for pupil detection - inference-aligned version.

This version trains on **raw images without preprocessing** (no gamma correction or CLAHE), so the model learns to handle raw camera input directly. This ensures the trained model works correctly at inference time without requiring preprocessing.

## Usage

```bash
# Install dependencies
uv pip install -e .

# Run training
train-ellipse-inf --epochs 15 --batch-size 4 --lr 0.001
```

## Arguments

- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.001)
- `--output-dir`: Output directory for checkpoints (default: ./checkpoints)
- `--device`: Device to use (auto/cuda/cpu, default: auto)
- `--no-mlflow`: Disable MLflow logging

## Difference from train-ellipse

The standard `train-ellipse` app uses preprocessing (gamma correction, CLAHE) which was applied to the training data. The `train-ellipse-inf` version trains on raw images, making it suitable for deployment where preprocessing may not be available.
