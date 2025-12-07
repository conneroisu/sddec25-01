## 1. Implementation

- [x] 1.1 Create `train_efficientvit_local.py` base file with imports (remove modal import)
- [x] 1.2 Add argparse CLI for configurable hyperparameters (epochs, batch_size, lr, data_dir, output_dir, etc.)
- [x] 1.3 Convert dataset loading to use local cache directory (replace Modal Volume with `~/.cache/openeds` default)
- [x] 1.4 Replace Modal secrets with environment variables for MLflow credentials (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_ID)
- [x] 1.5 Remove Modal decorators and move all code to main execution block with `if __name__ == "__main__"`
- [x] 1.6 Add device detection (CUDA/MPS/CPU) with `--device` CLI option for user override
- [x] 1.7 Add local model checkpoint saving to configurable output directory (default: `./checkpoints`)
- [x] 1.8 Preserve all model architecture code (TinyEfficientViT classes) unchanged
- [x] 1.9 Preserve all training logic (loss functions, metrics, visualization) unchanged

## 2. Validation

- [x] 2.1 Verify script runs without Modal installed (`python train_efficientvit_local.py --help`)
- [x] 2.2 Test dataset download and caching on local filesystem
- [x] 2.3 Verify training loop executes correctly on available device (GPU/CPU)
- [x] 2.4 Confirm MLflow logging works with environment variable credentials (or gracefully skips if not configured)
- [x] 2.5 Validate model checkpoint saving to local directory
- [x] 2.6 Test with reduced epochs (e.g., `--epochs 1`) for quick validation
