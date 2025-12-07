## ADDED Requirements

### Requirement: EfficientViT Model Package Structure
The system SHALL provide a Python package `packages/effvit/` that exports `TinyEfficientViTSeg` and related components for semantic segmentation.

#### Scenario: Package import
- **WHEN** user imports `from effvit import TinyEfficientViTSeg`
- **THEN** the model class SHALL be available
- **AND** model can be instantiated without additional imports

#### Scenario: Package structure
- **WHEN** package is inspected
- **THEN** it SHALL contain `model.py` with the complete segmentation model
- **AND** it SHALL contain `encoder.py` with encoder components
- **AND** it SHALL contain `decoder.py` with decoder component
- **AND** it SHALL contain `attention.py` with attention modules
- **AND** it SHALL contain `layers.py` with shared building blocks

### Requirement: TinyEfficientViTSeg Model Architecture
The system SHALL implement a `TinyEfficientViTSeg` model that performs semantic segmentation on grayscale eye images.

#### Scenario: Model instantiation
- **WHEN** `TinyEfficientViTSeg(in_channels=1, num_classes=2)` is created
- **THEN** model SHALL have a `TinyEfficientVitEncoder` backbone
- **AND** model SHALL have a `TinySegmentationDecoder` head
- **AND** model SHALL initialize weights using Kaiming normal

#### Scenario: Forward pass
- **WHEN** model receives input tensor of shape (B, 1, H, W)
- **THEN** output tensor SHALL have shape (B, num_classes, H, W)
- **AND** output SHALL be interpolated to match input spatial dimensions

#### Scenario: Default configuration
- **WHEN** model is instantiated with defaults
- **THEN** embed_dims SHALL be (8, 16, 24)
- **AND** depths SHALL be (1, 1, 1)
- **AND** num_heads SHALL be (1, 1, 2)
- **AND** decoder_dim SHALL be 16

### Requirement: TinyEfficientVitEncoder Component
The system SHALL implement a `TinyEfficientVitEncoder` that extracts multi-scale features using efficient vision transformer blocks.

#### Scenario: Encoder construction
- **WHEN** `TinyEfficientVitEncoder(in_channels, embed_dims, depths, num_heads, ...)` is created
- **THEN** it SHALL include a `TinyPatchEmbedding` for initial feature extraction
- **AND** it SHALL include 3 `TinyEfficientVitStage` modules

#### Scenario: Multi-scale features
- **WHEN** encoder processes input
- **THEN** it SHALL return tuple of 3 feature maps (f1, f2, f3)
- **AND** each feature map SHALL have progressively smaller spatial dimensions
- **AND** each feature map SHALL have increasing channel dimensions

### Requirement: TinySegmentationDecoder Component
The system SHALL implement a `TinySegmentationDecoder` with FPN-style skip connections.

#### Scenario: Decoder construction
- **WHEN** `TinySegmentationDecoder(encoder_dims, decoder_dim, num_classes)` is created
- **THEN** it SHALL have lateral connections for each encoder scale
- **AND** it SHALL have smoothing convolutions for each scale
- **AND** it SHALL have a final classification head

#### Scenario: Decoder forward
- **WHEN** decoder receives multi-scale features (f1, f2, f3) and target_size
- **THEN** it SHALL progressively upsample from deepest to shallowest
- **AND** it SHALL fuse features at each scale
- **AND** output SHALL be interpolated to target_size

### Requirement: TinyLocalWindowAttention Module
The system SHALL implement local window attention for efficient self-attention computation.

#### Scenario: Window partitioning
- **WHEN** attention processes input of shape (B, C, H, W)
- **THEN** it SHALL partition input into non-overlapping windows
- **AND** it SHALL apply attention within each window
- **AND** it SHALL restore original spatial layout after attention

#### Scenario: Padding handling
- **WHEN** input dimensions are not divisible by window_size
- **THEN** module SHALL pad input to nearest multiple
- **AND** module SHALL remove padding after attention
- **AND** output shape SHALL match input shape

### Requirement: TinyCascadedGroupAttention Module
The system SHALL implement cascaded group attention with efficient QKV computation.

#### Scenario: Attention computation
- **WHEN** attention receives input of shape (B, N, C)
- **THEN** it SHALL compute Q, K, V using single linear projection
- **AND** it SHALL apply scaled dot-product attention
- **AND** output shape SHALL be (B, N, C)

### Requirement: Shared Layer Components
The system SHALL provide reusable layer components for building the model.

#### Scenario: TinyConvNorm layer
- **WHEN** `TinyConvNorm(in_channels, out_channels, ...)` is created
- **THEN** it SHALL combine Conv2d and BatchNorm2d
- **AND** forward pass SHALL apply conv then batchnorm

#### Scenario: TinyPatchEmbedding layer
- **WHEN** `TinyPatchEmbedding(in_channels, embed_dim)` is created
- **THEN** it SHALL reduce spatial dimensions by 4x (stride 4 total)
- **AND** it SHALL use 2 convolutional stages with GELU activation

#### Scenario: TinyMLP layer
- **WHEN** `TinyMLP(dim, expansion_ratio)` is created
- **THEN** it SHALL have hidden_dim = dim * expansion_ratio
- **AND** it SHALL apply fc1 -> GELU -> fc2

### Requirement: Package Exports
The system SHALL export all public components from the package `__init__.py`.

#### Scenario: All exports available
- **WHEN** `from effvit import *` is executed
- **THEN** `TinyEfficientViTSeg` SHALL be available
- **AND** `TinyEfficientVitEncoder` SHALL be available
- **AND** `TinySegmentationDecoder` SHALL be available
- **AND** `TinyCascadedGroupAttention` SHALL be available
- **AND** `TinyLocalWindowAttention` SHALL be available
- **AND** `TinyConvNorm` SHALL be available
- **AND** `TinyPatchEmbedding` SHALL be available
- **AND** `TinyMLP` SHALL be available
- **AND** `TinyEfficientVitBlock` SHALL be available
- **AND** `TinyEfficientVitStage` SHALL be available
