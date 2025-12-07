## ADDED Requirements

### Requirement: EfficientViT Segmentation Colab Notebook
The system SHALL provide a Jupyter notebook `train_efficientvit.ipynb` that trains a TinyEfficientViTSeg model for eye pupil segmentation, runnable in Google Colab with free GPU.

#### Scenario: Notebook execution in Colab
- **WHEN** user opens `training/train_efficientvit.ipynb` in Google Colab
- **AND** user selects GPU runtime
- **AND** user runs all cells
- **THEN** training completes successfully
- **AND** trained model is saved as PyTorch checkpoint

#### Scenario: Dataset loading from HuggingFace
- **WHEN** notebook executes dataset loading cells
- **THEN** OpenEDS dataset SHALL be downloaded from `Conner/openeds-precomputed`
- **AND** train and validation splits SHALL be available

#### Scenario: Model architecture consistency
- **WHEN** TinyEfficientViTSeg is instantiated in notebook
- **THEN** model architecture SHALL match `train_efficientvit.py` Modal version
- **AND** model SHALL have less than 60,000 trainable parameters
- **AND** model SHALL output shape (B, 2, 400, 640) for input shape (B, 1, 400, 640)

#### Scenario: Training visualization
- **WHEN** training completes
- **THEN** notebook SHALL display loss curves (total, CE, dice, surface)
- **AND** notebook SHALL display sample prediction visualizations
- **AND** notebook SHALL report final mIoU and per-class IoU

### Requirement: EfficientViT Model Export
The EfficientViT notebook SHALL provide functionality to export and download the trained segmentation model.

#### Scenario: PyTorch checkpoint save
- **WHEN** training completes with best validation mIoU
- **THEN** notebook SHALL save model to `best_efficientvit_model.pt`
- **AND** notebook SHALL display file size

#### Scenario: ONNX export
- **WHEN** user runs the ONNX export cell
- **THEN** notebook SHALL export model to `best_efficientvit_model.onnx`
- **AND** ONNX file size SHALL be less than 300KB

#### Scenario: Model download
- **WHEN** user runs the download cell
- **THEN** trained model files SHALL be downloadable from Colab
