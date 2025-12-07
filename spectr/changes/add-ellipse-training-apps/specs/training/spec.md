## ADDED Requirements

### Requirement: Ellipse Training Application
The system SHALL provide a training application `apps/train_ellipse/` that trains an `EllipseRegressionNet` model for pupil ellipse parameter prediction, using the model from `packages/ellipse/`.

#### Scenario: CLI execution
- **WHEN** user runs `train-ellipse --epochs 15 --batch-size 4`
- **THEN** training executes using specified hyperparameters
- **AND** model is imported from `ellipse` package
- **AND** checkpoints are saved to specified output directory

#### Scenario: Dataset loading
- **WHEN** training loads dataset
- **THEN** it SHALL load from `Conner/sddec25-01` HuggingFace repository
- **AND** it SHALL use precomputed ellipse parameters (cx, cy, rx, ry) from dataset
- **AND** it SHALL apply augmentations (line augment, gaussian blur, horizontal flip)

#### Scenario: Model import from package
- **WHEN** training script initializes model
- **THEN** it SHALL import `EllipseRegressionNet` from `ellipse` package
- **AND** it SHALL import `EllipseRegressionLoss` from `ellipse` package
- **AND** no model definition code SHALL exist in the training app

#### Scenario: MLflow integration
- **WHEN** MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID are set
- **THEN** training SHALL log metrics, parameters, and artifacts to MLflow
- **AND** model_type tag SHALL be "EllipseRegressionNet"

### Requirement: Inference-Aligned Ellipse Training Application
The system SHALL provide a training application `apps/train_ellipse_inf/` that trains on raw images without preprocessing, so the trained model works directly on camera input.

#### Scenario: CLI execution
- **WHEN** user runs `train-ellipse-inf --epochs 15 --batch-size 4`
- **THEN** training executes on raw images (no gamma correction, no CLAHE)
- **AND** model is imported from `ellipse` package

#### Scenario: Raw image training
- **WHEN** training loads images from dataset
- **THEN** images SHALL be used directly without preprocessing
- **AND** only stochastic augmentations (line, blur, flip) SHALL be applied
- **AND** model learns to handle raw camera input

#### Scenario: Dataset source
- **WHEN** inference-aligned training loads dataset
- **THEN** it SHALL load from `Conner/sddec25-01` HuggingFace repository
- **AND** it SHALL use the same precomputed ellipse parameters as standard training

### Requirement: Ellipse Training App pyproject.toml Configuration
The training applications SHALL follow the established monorepo pattern with workspace dependencies.

#### Scenario: Standard app dependencies
- **WHEN** `apps/train_ellipse/pyproject.toml` is inspected
- **THEN** it SHALL include `ellipse` as workspace dependency
- **AND** it SHALL include all required training dependencies (torch, datasets, mlflow, etc.)
- **AND** it SHALL define CLI entry point `train-ellipse = "train_ellipse.main:main"`

#### Scenario: Inference-aligned app dependencies
- **WHEN** `apps/train_ellipse_inf/pyproject.toml` is inspected
- **THEN** it SHALL include `ellipse` as workspace dependency
- **AND** it SHALL define CLI entry point `train-ellipse-inf = "train_ellipse_inf.main:main"`

### Requirement: GPU Training Optimizations for Ellipse Apps
The ellipse training applications SHALL include GPU optimizations consistent with existing training apps.

#### Scenario: Memory format optimization
- **WHEN** training runs on CUDA device
- **THEN** model SHALL use channels_last memory format
- **AND** model SHALL optionally use torch.compile

#### Scenario: Mixed precision training
- **WHEN** training runs on CUDA device
- **THEN** torch.amp.autocast SHALL be used for forward passes
- **AND** GradScaler SHALL be used for gradient scaling

#### Scenario: GPU metrics accumulation
- **WHEN** training computes epoch metrics
- **THEN** metrics SHALL be accumulated on GPU
- **AND** GPU-to-CPU transfer SHALL occur at end of epoch
