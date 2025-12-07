## ADDED Requirements

### Requirement: Local EfficientViT Training Script
The system SHALL provide a local training script `train_efficientvit_local.py` that trains a TinyEfficientViT model for eye pupil segmentation on the OpenEDS dataset without requiring Modal cloud infrastructure.

#### Scenario: Local training script execution
- **WHEN** user runs `python training/train_efficientvit_local.py`
- **THEN** training executes on the local machine
- **AND** uses local filesystem for dataset caching
- **AND** saves model checkpoints to local directory

#### Scenario: CLI argument configuration
- **WHEN** user runs `python training/train_efficientvit_local.py --epochs 10 --batch-size 64`
- **THEN** training uses the specified hyperparameters
- **AND** defaults are used for unspecified arguments

#### Scenario: Device selection
- **WHEN** user runs `python training/train_efficientvit_local.py --device cuda`
- **THEN** training runs on the specified device
- **AND** if device is unavailable, script exits with clear error message

#### Scenario: MLflow credentials via environment
- **WHEN** MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID environment variables are set
- **THEN** training logs metrics to the configured MLflow server
- **WHEN** MLflow environment variables are not set
- **THEN** training continues without MLflow logging and prints warning

#### Scenario: Local dataset caching
- **WHEN** dataset is not cached locally
- **THEN** script downloads from HuggingFace and caches to `--data-dir` (default: `~/.cache/openeds`)
- **WHEN** dataset is already cached
- **THEN** script loads from local cache without re-downloading

#### Scenario: Model checkpoint output
- **WHEN** training completes an epoch with improved validation mIoU
- **THEN** best model is saved to `--output-dir` (default: `./checkpoints`)
- **AND** checkpoint filename includes epoch number and validation mIoU
