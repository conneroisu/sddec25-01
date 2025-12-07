## ADDED Requirements

### Requirement: Local Ellipse Regression Training Script
The system SHALL provide a local training script `train_ellipse_local.py` that trains the EllipseRegressionNet model without requiring Modal cloud infrastructure.

#### Scenario: Run training with default parameters
- **WHEN** user executes `python train_ellipse_local.py`
- **THEN** the script downloads the dataset from HuggingFace (if not cached locally)
- **AND** trains the EllipseRegressionNet model for the default number of epochs
- **AND** saves model checkpoints to the local filesystem
- **AND** logs metrics to MLflow (if credentials configured)

#### Scenario: Run training with custom hyperparameters
- **WHEN** user executes `python train_ellipse_local.py --epochs 20 --batch-size 16 --lr 0.0005`
- **THEN** the script uses the specified hyperparameters for training
- **AND** overrides the default values

#### Scenario: Specify local data directory
- **WHEN** user executes `python train_ellipse_local.py --data-dir /path/to/cache`
- **THEN** the script uses the specified directory for dataset caching
- **AND** skips download if dataset already exists at that location

#### Scenario: Run without MLflow credentials
- **WHEN** user executes the script without `MLFLOW_TRACKING_URI` environment variable
- **THEN** the script runs training without MLflow logging
- **AND** prints a warning about disabled MLflow logging
- **AND** still saves model checkpoints locally

### Requirement: Local Device Detection
The local training script SHALL automatically detect and use available compute devices.

#### Scenario: CUDA GPU available
- **WHEN** the system has a CUDA-capable GPU
- **THEN** training runs on the GPU by default

#### Scenario: CPU-only system
- **WHEN** no CUDA GPU is available
- **THEN** training falls back to CPU
- **AND** prints a warning about expected slower performance

#### Scenario: User forces CPU execution
- **WHEN** user executes `python train_ellipse_local.py --device cpu`
- **THEN** training runs on CPU regardless of GPU availability

### Requirement: Local Dataset Caching
The local training script SHALL cache the dataset to avoid repeated downloads.

#### Scenario: First run downloads dataset
- **WHEN** the dataset is not present in the cache directory
- **THEN** the script downloads from HuggingFace Hub
- **AND** saves to the local cache directory
- **AND** prints download progress

#### Scenario: Subsequent runs use cache
- **WHEN** the dataset exists in the cache directory
- **THEN** the script loads from local cache without network access
- **AND** prints confirmation of cache usage
