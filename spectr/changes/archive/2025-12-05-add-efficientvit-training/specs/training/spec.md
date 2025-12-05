## ADDED Requirements

### Requirement: EfficientViT Training Script
The system SHALL provide a training script `train_efficientvit.py` that trains a TinyEfficientViT model for eye pupil segmentation on the OpenEDS dataset.

#### Scenario: Training script execution
- **WHEN** user runs `modal run training/train_efficientvit.py`
- **THEN** training completes and logs metrics to MLflow
- **AND** best model is exported to ONNX format

#### Scenario: Model parameter constraint
- **WHEN** TinyEfficientViTSeg model is instantiated
- **THEN** total trainable parameters SHALL be less than 60,000

### Requirement: TinyEfficientViT Model Architecture
The system SHALL implement a TinyEfficientViTSeg model based on EfficientViT-MSRA architecture with drastically reduced dimensions to meet the <60k parameter constraint.

#### Scenario: Model configuration
- **WHEN** TinyEfficientViTSeg is created with default config
- **THEN** embed_dim SHALL be approximately (8, 16, 24)
- **AND** depth SHALL be (1, 1, 1)
- **AND** num_heads SHALL be (1, 1, 2)

#### Scenario: Forward pass
- **WHEN** model receives input tensor of shape (B, 1, 400, 640)
- **THEN** output tensor SHALL have shape (B, 2, 400, 640)

### Requirement: Segmentation Decoder
The system SHALL implement a lightweight decoder that produces dense predictions from multi-scale encoder features.

#### Scenario: Decoder with skip connections
- **WHEN** encoder produces features at multiple scales
- **THEN** decoder SHALL upsample and combine features to produce full-resolution output

### Requirement: ONNX Export
The system SHALL export the trained model to ONNX format compatible with edge deployment.

#### Scenario: ONNX export
- **WHEN** training completes with best validation mIoU
- **THEN** best model SHALL be exported to `best_efficientvit_model.onnx`
- **AND** file size SHALL be less than 300KB

### Requirement: MLflow Integration
The system SHALL log training metrics, parameters, and artifacts to MLflow.

#### Scenario: Metric logging
- **WHEN** each epoch completes
- **THEN** train_loss, valid_loss, train_iou, valid_iou SHALL be logged
- **AND** loss components (ce_loss, dice_loss, surface_loss) SHALL be logged

#### Scenario: Model tagging
- **WHEN** MLflow run is created
- **THEN** model_type tag SHALL be "TinyEfficientViT"
- **AND** architecture tag SHALL distinguish from ShallowNet runs
