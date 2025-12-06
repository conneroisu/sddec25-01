## ADDED Requirements

### Requirement: Dataset Precompute Script
The system SHALL provide a precompute script `training/precompute.py` that downloads the raw OpenEDS dataset from Kaggle, applies deterministic preprocessing (gamma correction and CLAHE), and pushes the preprocessed dataset to HuggingFace.

#### Scenario: Script execution
- **WHEN** user runs `python training/precompute.py`
- **THEN** OpenEDS dataset SHALL be downloaded from Kaggle
- **AND** gamma correction (gamma=0.8) SHALL be applied to all images
- **AND** CLAHE (clipLimit=1.5, tileGridSize=8x8) SHALL be applied to all images
- **AND** preprocessed dataset SHALL be pushed to HuggingFace

#### Scenario: Kaggle authentication
- **WHEN** user runs precompute script
- **AND** Kaggle API credentials are configured in `~/.kaggle/kaggle.json`
- **THEN** script SHALL authenticate and download datasets
- **AND** script SHALL combine train and validation splits

#### Scenario: Dataset columns
- **WHEN** preprocessed dataset is created
- **THEN** dataset SHALL contain `image` column with uint8 preprocessed images
- **AND** dataset SHALL contain `label` column with segmentation masks
- **AND** dataset SHALL contain `spatial_weights` column with edge weights
- **AND** dataset SHALL contain `dist_map` column with signed distance maps
- **AND** dataset SHALL contain `preprocessed` column set to True

#### Scenario: HuggingFace upload
- **WHEN** preprocessing completes
- **THEN** dataset SHALL be pushed to `Conner/openeds-precomputed`
- **AND** upload SHALL support resume on failure

### Requirement: Precompute CLI Arguments
The precompute script SHALL support command-line arguments for configuration.

#### Scenario: Help output
- **WHEN** user runs `python training/precompute.py --help`
- **THEN** script SHALL display usage information with all available options

#### Scenario: HuggingFace repository override
- **WHEN** user runs `python training/precompute.py --hf-repo MyOrg/my-dataset`
- **THEN** script SHALL push to the specified HuggingFace repository

#### Scenario: Local output option
- **WHEN** user runs `python training/precompute.py --output-dir ./local-dataset --no-push`
- **THEN** script SHALL save preprocessed dataset locally
- **AND** script SHALL NOT push to HuggingFace

#### Scenario: Validation only option
- **WHEN** user runs `python training/precompute.py --validate`
- **THEN** script SHALL download and validate dataset without preprocessing
- **AND** script SHALL report sample counts and data statistics

## MODIFIED Requirements

### Requirement: EfficientViT Training Script
The system SHALL provide a training script `train_efficientvit.py` that trains a TinyEfficientViT model for eye pupil segmentation on the OpenEDS dataset.

#### Scenario: Training script execution
- **WHEN** user runs `modal run training/train_efficientvit.py`
- **THEN** training completes and logs metrics to MLflow
- **AND** best model is exported to ONNX format

#### Scenario: Model parameter constraint
- **WHEN** TinyEfficientViTSeg model is instantiated
- **THEN** total trainable parameters SHALL be less than 60,000

#### Scenario: Preprocessed dataset detection
- **WHEN** dataset contains `preprocessed` column set to True
- **THEN** IrisDataset SHALL skip gamma correction and CLAHE
- **AND** IrisDataset SHALL load images directly from dataset

#### Scenario: Legacy dataset compatibility
- **WHEN** dataset does not contain `preprocessed` column
- **THEN** IrisDataset SHALL apply gamma correction (gamma=0.8)
- **AND** IrisDataset SHALL apply CLAHE (clipLimit=1.5, tileGridSize=8x8)
