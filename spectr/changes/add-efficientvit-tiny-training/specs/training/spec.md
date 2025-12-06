## ADDED Requirements

### Requirement: Ultra-Tiny EfficientViT Training Script
The system SHALL provide a training script `train_efficientvit_tiny_local.py` that trains a TinyEfficientViT-Micro model with less than 10,000 parameters for eye pupil segmentation on the OpenEDS dataset.

#### Scenario: Training script execution
- **WHEN** user runs `python training/train_efficientvit_tiny_local.py`
- **THEN** training completes and logs metrics to MLflow
- **AND** best model is exported to ONNX format

#### Scenario: Model parameter constraint
- **WHEN** TinyEfficientViT-Micro model is instantiated
- **THEN** total trainable parameters SHALL be less than 10,000

### Requirement: TinyEfficientViT-Micro Model Architecture
The system SHALL implement a TinyEfficientViT-Micro model with drastically reduced dimensions to meet the <10k parameter constraint.

#### Scenario: Model configuration
- **WHEN** TinyEfficientViT-Micro is created with default config
- **THEN** embed_dim SHALL be (8, 12, 18)
- **AND** decoder_dim SHALL be 8
- **AND** num_heads SHALL be (1, 1, 1)

#### Scenario: Forward pass
- **WHEN** model receives input tensor of shape (B, 1, 400, 640)
- **THEN** output tensor SHALL have shape (B, 2, 400, 640)

### Requirement: MLflow Tagging for Micro Variant
The system SHALL tag MLflow runs to distinguish TinyEfficientViT-Micro from other model variants.

#### Scenario: Model tagging
- **WHEN** MLflow run is created for micro model training
- **THEN** model_type tag SHALL be "TinyEfficientViT-Micro"
- **AND** tag SHALL distinguish from standard TinyEfficientViT runs
