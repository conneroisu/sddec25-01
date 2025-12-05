## ADDED Requirements

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
- **THEN** OpenEDS dataset SHALL be downloaded from `Conner/openeds-precomputed`
- **AND** train and validation splits SHALL be available

#### Scenario: Model architecture consistency
- **WHEN** EllipseRegressionNet is instantiated in notebook
- **THEN** model architecture SHALL match `train_ellipse.py` Modal version
- **AND** model SHALL output 4 parameters (cx, cy, rx, ry)

#### Scenario: Training visualization
- **WHEN** training completes
- **THEN** notebook SHALL display loss curves
- **AND** notebook SHALL display sample prediction visualizations
- **AND** notebook SHALL report final mIoU and error metrics

### Requirement: Colab Runtime Configuration
The notebook SHALL include setup cells that configure the Colab environment for GPU training.

#### Scenario: GPU availability check
- **WHEN** user runs the setup cells
- **THEN** notebook SHALL verify GPU is available
- **AND** notebook SHALL print GPU name and memory

#### Scenario: Dependency installation
- **WHEN** user runs the pip install cell
- **THEN** required packages SHALL be installed (torch, torchvision, opencv-python, datasets, pillow, scikit-learn, tqdm, matplotlib, onnx)

### Requirement: Model Export from Colab
The notebook SHALL provide functionality to export and download the trained model.

#### Scenario: ONNX export
- **WHEN** training completes with best validation mIoU
- **THEN** notebook SHALL export model to `best_ellipse_model.onnx`
- **AND** notebook SHALL display file size

#### Scenario: Model download
- **WHEN** user runs the download cell
- **THEN** trained ONNX model SHALL be downloadable from Colab
