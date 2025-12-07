## ADDED Requirements

### Requirement: Ultra-Tiny EfficientViT Training Script
The system SHALL provide a training script `train_efficientvit_tiny_local.py` that trains a TinyEfficientViT-Micro model with less than 10,000 parameters for eye pupil segmentation on the OpenEDS dataset.

#### Scenario: Training script execution
- **WHEN** user runs `python training/train_efficientvit_tiny_local.py`
- **THEN** training completes and logs metrics to MLflow
- **AND** best model is saved to `best_efficientvit_tiny_model.pt`

#### Scenario: Distinct checkpoint filenames
- **WHEN** training saves model checkpoints
- **THEN** best model SHALL be saved as `best_efficientvit_tiny_model.pt`
- **AND** epoch checkpoints SHALL be saved as `efficientvit_tiny_model_epoch_{n}.pt`
- **AND** filenames SHALL NOT conflict with standard EfficientViT checkpoints

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

### Requirement: GPU Training Optimizations
The system SHALL include all GPU optimizations from the standard EfficientViT training script.

#### Scenario: Memory format optimization
- **WHEN** training runs on CUDA device
- **THEN** model SHALL use channels_last memory format
- **AND** torch.compile with max-autotune mode SHALL be applied

#### Scenario: Mixed precision training
- **WHEN** training runs on CUDA device
- **THEN** torch.amp.autocast SHALL be used for forward passes
- **AND** GradScaler SHALL be used for gradient scaling

#### Scenario: CUDA optimizations
- **WHEN** training runs on CUDA device
- **THEN** cudnn.benchmark SHALL be enabled
- **AND** TF32 SHALL be enabled for matmul and cudnn

#### Scenario: GPU metrics accumulation
- **WHEN** training computes epoch metrics
- **THEN** all metrics SHALL be accumulated on GPU
- **AND** GPU-to-CPU transfer SHALL occur only at end of training
- **AND** pre-allocated GPU tensors SHALL be used for metric storage

#### Scenario: Efficient data loading
- **WHEN** DataLoaders are created for CUDA training
- **THEN** pin_memory SHALL be enabled
- **AND** non_blocking transfers SHALL be used

### Requirement: MLflow Tagging for Micro Variant
The system SHALL tag MLflow runs to distinguish TinyEfficientViT-Micro from other model variants.

#### Scenario: Model tagging
- **WHEN** MLflow run is created for micro model training
- **THEN** model_type tag SHALL be "TinyEfficientViT-Micro"
- **AND** tag SHALL distinguish from standard TinyEfficientViT runs
