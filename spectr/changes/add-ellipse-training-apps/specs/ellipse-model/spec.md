# Ellipse Model Specification

## Purpose

Define requirements for the ellipse regression model package that provides reusable model architecture and utilities for pupil ellipse parameter prediction.

## ADDED Requirements

### Requirement: Ellipse Model Package Structure
The system SHALL provide a Python package `packages/ellipse/` that exports `EllipseRegressionNet` and related utilities for ellipse regression.

#### Scenario: Package import
- **WHEN** user imports `from ellipse import EllipseRegressionNet`
- **THEN** the model class SHALL be available
- **AND** model can be instantiated without additional imports

#### Scenario: Package structure
- **WHEN** package is inspected
- **THEN** it SHALL contain `model.py` with network architecture
- **AND** it SHALL contain `loss.py` with loss function
- **AND** it SHALL contain `metrics.py` with IoU and ellipse utilities

### Requirement: EllipseRegressionNet Architecture
The system SHALL implement an `EllipseRegressionNet` model that predicts ellipse parameters (cx, cy, rx, ry) from grayscale eye images.

#### Scenario: Model instantiation
- **WHEN** `EllipseRegressionNet(in_channels=1, channel_size=32, dropout=True, prob=0.2)` is created
- **THEN** model SHALL have 4 `DownBlock` encoder stages
- **AND** model SHALL output 4 normalized parameters via sigmoid activation

#### Scenario: Forward pass
- **WHEN** model receives input tensor of shape (B, 1, 400, 640)
- **THEN** output tensor SHALL have shape (B, 4)
- **AND** output values SHALL be in range [0, 1] (normalized)

#### Scenario: Parameter count
- **WHEN** model is instantiated with default channel_size=32
- **THEN** total trainable parameters SHALL be consistent with `training/train_ellipse_local.py`

### Requirement: DownBlock Encoder Component
The system SHALL implement a `DownBlock` module using depthwise separable convolutions for efficient feature extraction.

#### Scenario: DownBlock construction
- **WHEN** `DownBlock(input_channels, output_channels, down_size=(2,2))` is created
- **THEN** it SHALL include depthwise and pointwise convolutions
- **AND** it SHALL apply batch normalization
- **AND** it SHALL support optional dropout

#### Scenario: Dense connections
- **WHEN** `DownBlock` processes input
- **THEN** it SHALL use dense connections (concatenation) between conv layers
- **AND** final output SHALL be batch-normalized

### Requirement: EllipseRegressionLoss Function
The system SHALL implement an `EllipseRegressionLoss` that combines center and radius losses.

#### Scenario: Loss computation
- **WHEN** loss is computed with `criterion(pred, target)`
- **THEN** it SHALL return tuple (total_loss, center_loss, radius_loss)
- **AND** total_loss SHALL be weighted combination of center and radius losses

#### Scenario: Loss components
- **WHEN** loss is initialized with `center_weight=1.0, radius_weight=1.0, iou_weight=0.5`
- **THEN** center_loss SHALL use SmoothL1 on pred[:, :2] vs target[:, :2]
- **AND** radius_loss SHALL use SmoothL1 on pred[:, 2:] vs target[:, 2:]
- **AND** optional IoU proxy loss SHALL be computed from parameter MSE

### Requirement: Ellipse Parameter Utilities
The system SHALL provide utility functions for normalizing, denormalizing, and rendering ellipse parameters.

#### Scenario: Parameter normalization
- **WHEN** `normalize_ellipse_params(cx, cy, rx, ry)` is called
- **THEN** cx SHALL be divided by IMAGE_WIDTH (640)
- **AND** cy SHALL be divided by IMAGE_HEIGHT (400)
- **AND** rx, ry SHALL be divided by MAX_RADIUS (sqrt(640^2 + 400^2)/2)

#### Scenario: Parameter denormalization
- **WHEN** `denormalize_ellipse_params(cx_norm, cy_norm, rx_norm, ry_norm)` is called
- **THEN** it SHALL reverse the normalization process
- **AND** return pixel coordinates and radii

#### Scenario: Ellipse mask rendering
- **WHEN** `render_ellipse_mask(cx, cy, rx, ry)` is called
- **THEN** it SHALL return a (400, 640) binary mask
- **AND** mask SHALL have 1s inside the ellipse, 0s outside

### Requirement: GPU-Accelerated IoU Computation
The system SHALL provide GPU-optimized functions for computing IoU between predicted and ground truth ellipse masks.

#### Scenario: GPU IoU computation
- **WHEN** `compute_iou_with_gt_mask_gpu(pred, gt_masks, device)` is called
- **THEN** predicted ellipse masks SHALL be rendered on GPU
- **AND** IoU SHALL be computed without CPU data transfer
- **AND** function SHALL return (mean_iou, bg_iou, pupil_iou)

#### Scenario: Ellipse mask GPU rendering
- **WHEN** `render_ellipse_mask_gpu(pred_params)` is called with (batch, 4) tensor
- **THEN** it SHALL return (batch, H, W) mask tensor on same device
- **AND** computation SHALL use vectorized ellipse equation
