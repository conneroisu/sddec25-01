## Context

The project follows a monorepo structure with:
- `packages/` - Reusable model definitions and utilities
- `apps/` - Training applications with CLI entry points
- `training/` - Legacy training scripts (Modal-based and local)

The TinyEfficientViT model already follows this pattern:
- `packages/tiny_effvit/` - Model definition only
- `apps/train_tiny_effvit/` - Standard training with preprocessing
- `apps/train_tiny_effvit_inf/` - Inference-aligned training (raw images)

## Goals / Non-Goals

**Goals:**
- Extract `EllipseRegressionNet` model to reusable package
- Create training apps following established pattern
- Maintain functional parity with `training/train_ellipse_local.py`
- Support both preprocessed and raw image training modes

**Non-Goals:**
- Deprecating existing `training/train_ellipse_local.py` (remains as reference)
- Changing model architecture or hyperparameters
- Adding new features beyond refactoring

## Decisions

### Package Structure
- **Decision**: Create `packages/ellipse/ellipse/` with model.py, loss.py, metrics.py modules
- **Rationale**: Follows tiny_effvit pattern; separates concerns clearly
- **Alternatives**: Single file (rejected - less maintainable), nested classes (rejected - harder to import)

### Model Components to Extract
1. `EllipseRegressionNet` - Main regression model
2. `DownBlock` - Encoder building block
3. `EllipseRegressionLoss` - Custom loss function
4. Helper functions: `normalize_ellipse_params`, `denormalize_ellipse_params`, `render_ellipse_mask`
5. GPU IoU computation functions

### Training App Split
- **Standard app** (`train_ellipse`): Uses precomputed ellipse parameters from dataset, applies augmentations
- **Inference-aligned app** (`train_ellipse_inf`): Trains on raw images without preprocessing, uses `Conner/sddec25-01` dataset

### pyproject.toml Pattern
- Use workspace dependencies: `ellipse = { workspace = true }`
- Include all training dependencies (torch, datasets, mlflow, etc.)
- Define CLI entry points: `train-ellipse = "train_ellipse.main:main"`

## Risks / Trade-offs

- **Risk**: Model behavior divergence between package and original script
  - **Mitigation**: Extract exact code from train_ellipse_local.py without modifications
- **Risk**: Import path changes breaking downstream code
  - **Mitigation**: Package not yet used elsewhere; apps are new consumers

## Migration Plan

1. Create packages/ellipse model definition
2. Create apps/train_ellipse importing from package
3. Create apps/train_ellipse_inf importing from package
4. Validate both apps produce identical results to original script
5. No changes to training/train_ellipse_local.py (kept as reference)

## Open Questions

- None - following established patterns exactly
