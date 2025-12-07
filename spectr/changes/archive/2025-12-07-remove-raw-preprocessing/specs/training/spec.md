## MODIFIED Requirements

### Requirement: EfficientViT Training Script
The system SHALL provide a training script `train_efficientvit.py` that trains a TinyEfficientViT model for eye pupil segmentation on the preprocessed OpenEDS dataset.

#### Scenario: Training script execution
- **WHEN** user runs `modal run training/train_efficientvit.py`
- **THEN** training completes using preprocessed dataset from HuggingFace
- **AND** logs metrics to MLflow
- **AND** best model is exported to ONNX format

#### Scenario: Model parameter constraint
- **WHEN** TinyEfficientViTSeg model is instantiated
- **THEN** total trainable parameters SHALL be less than 60,000

#### Scenario: Dataset loading
- **WHEN** training script loads dataset
- **THEN** script SHALL load from `Conner/sddec25-01` HuggingFace repository
- **AND** script SHALL use preprocessed images directly (no runtime gamma/CLAHE)
- **AND** script SHALL use precomputed spatial_weights and dist_map

### Requirement: Ellipse Regression Colab Notebook
The system SHALL provide a Jupyter notebook `train_ellipse.ipynb` that trains an EllipseRegressionNet model for pupil ellipse parameter prediction, runnable in Google Colab with free GPU.

#### Scenario: Notebook execution in Colab
- **WHEN** user opens `training/train_ellipse.ipynb` in Google Colab
- **AND** user selects GPU runtime
- **AND** user runs all cells
- **THEN** training completes successfully
- **AND** trained model is exported to ONNX format

#### Scenario: Dataset loading from HuggingFace
- **WHEN** notebook executes dataset loading cells
- **THEN** OpenEDS dataset SHALL be downloaded from `Conner/sddec25-01`
- **AND** train and validation splits SHALL be available
- **AND** precomputed ellipse parameters (cx, cy, rx, ry) SHALL be available

#### Scenario: Model architecture consistency
- **WHEN** EllipseRegressionNet is instantiated in notebook
- **THEN** model architecture SHALL match `train_ellipse.py` Modal version
- **AND** model SHALL output 4 parameters (cx, cy, rx, ry)

#### Scenario: Training visualization
- **WHEN** training completes
- **THEN** notebook SHALL display loss curves
- **AND** notebook SHALL display sample prediction visualizations
- **AND** notebook SHALL report final mIoU and error metrics
