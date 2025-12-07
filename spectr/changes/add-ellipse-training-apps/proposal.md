# Change: Add Ellipse Training Applications and Model Package

## Why
The codebase has training scripts for TinyEfficientViT segmentation models (`apps/train_tiny_effvit` and `apps/train_tiny_effvit_inf`) that follow a clean separation pattern: model definition in `packages/tiny_effvit` and training apps in `apps/`. The ellipse regression model in `training/train_ellipse_local.py` currently embeds the model definition within the training script, which:
- Prevents model reuse across different training configurations
- Makes it harder to maintain consistent model architecture
- Doesn't follow the established monorepo pattern

This change establishes the same clean separation for ellipse regression models.

## What Changes
- **ADDED**: `packages/ellipse/` - Complete model definition package with `EllipseRegressionNet` and helper functions
- **ADDED**: `apps/train_ellipse/` - Standard ellipse training application (with preprocessing in dataset)
- **ADDED**: `apps/train_ellipse_inf/` - Inference-aligned ellipse training (raw images, no preprocessing)
- **MODIFIED**: `packages/ellipse/pyproject.toml` - Update to export model classes
- Follows existing patterns from `packages/tiny_effvit/` and `apps/train_tiny_effvit*/`

## Impact
- Affected specs: `training`, `ellipse-model` (new capability)
- Affected code:
  - `packages/ellipse/` - New model definition files
  - `apps/train_ellipse/` - New training application
  - `apps/train_ellipse_inf/` - New inference-aligned training application
- No breaking changes - existing `training/train_ellipse_local.py` remains functional
