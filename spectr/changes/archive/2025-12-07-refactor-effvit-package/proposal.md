# Change: Refactor EfficientViT Model into Dedicated Package

## Why

The `packages/tiny_effvit/` package currently contains the `TinyEfficientViTSeg` model but the package name doesn't follow the established naming pattern used by `packages/ellipse/`. Additionally, the model architecture needs to be separated into logical components (encoder, decoder, utilities) similar to how `packages/ellipse/` is organized with `model.py`, `loss.py`, and `metrics.py`.

This refactoring:
- Aligns package naming with the project convention (`effvit` instead of `tiny_effvit`)
- Improves code organization by splitting the monolithic `main.py` into focused modules
- Enables better reusability of individual components (encoder, decoder, attention blocks)
- Follows the pattern established by the successful `ellipse` package refactoring

## What Changes

- Create new `packages/effvit/` package with proper module structure:
  - `effvit/model.py` - Core model (`TinyEfficientViTSeg`)
  - `effvit/encoder.py` - Encoder components (`TinyEfficientVitEncoder`, `TinyEfficientVitStage`, `TinyEfficientVitBlock`)
  - `effvit/decoder.py` - Decoder component (`TinySegmentationDecoder`)
  - `effvit/attention.py` - Attention modules (`TinyCascadedGroupAttention`, `TinyLocalWindowAttention`)
  - `effvit/layers.py` - Shared building blocks (`TinyConvNorm`, `TinyPatchEmbedding`, `TinyMLP`)
  - `effvit/__init__.py` - Clean exports for all public components
- Remove `packages/tiny_effvit/` (deprecated)
- Update dependent apps to import from `effvit` instead of `tiny_effvit`:
  - `apps/demo_tiny_effvit/`
  - `apps/train_tiny_effvit/`
  - `apps/train_tiny_effvit_inf/`
- Update workspace configuration in root `pyproject.toml`

## Impact

- Affected specs: New `effvit-model` spec (following `ellipse-model` pattern)
- Affected code:
  - `packages/tiny_effvit/` - removed
  - `packages/effvit/` - new package
  - `apps/demo_tiny_effvit/pyproject.toml` - dependency update
  - `apps/train_tiny_effvit/pyproject.toml` - dependency update
  - `apps/train_tiny_effvit_inf/pyproject.toml` - dependency update
  - Root `pyproject.toml` - workspace member update
- **BREAKING**: Import path changes from `from tiny_effvit import ...` to `from effvit import ...`
