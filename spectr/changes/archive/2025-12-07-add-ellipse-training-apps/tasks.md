## 1. Create Ellipse Model Package

- [x] 1.1 Create `packages/ellipse/ellipse/__init__.py` exporting `EllipseRegressionNet` and helper functions
- [x] 1.2 Create `packages/ellipse/ellipse/model.py` with `EllipseRegressionNet`, `DownBlock`, and weight initialization
- [x] 1.3 Create `packages/ellipse/ellipse/loss.py` with `EllipseRegressionLoss` class
- [x] 1.4 Create `packages/ellipse/ellipse/metrics.py` with IoU computation and ellipse parameter functions
- [x] 1.5 Update `packages/ellipse/pyproject.toml` to properly package the module

## 2. Create Standard Ellipse Training App

- [x] 2.1 Create `apps/train_ellipse/` directory structure following `train_tiny_effvit` pattern
- [x] 2.2 Create `apps/train_ellipse/train_ellipse/__init__.py`
- [x] 2.3 Create `apps/train_ellipse/train_ellipse/main.py` with training logic (imports model from packages/ellipse)
- [x] 2.4 Create `apps/train_ellipse/pyproject.toml` with dependencies and CLI entry point
- [x] 2.5 Create `apps/train_ellipse/README.md` with usage instructions

## 3. Create Inference-Aligned Ellipse Training App

- [x] 3.1 Create `apps/train_ellipse_inf/` directory structure following `train_tiny_effvit_inf` pattern
- [x] 3.2 Create `apps/train_ellipse_inf/train_ellipse_inf/__init__.py`
- [x] 3.3 Create `apps/train_ellipse_inf/train_ellipse_inf/main.py` with raw image training (no preprocessing)
- [x] 3.4 Create `apps/train_ellipse_inf/pyproject.toml` with dependencies and CLI entry point
- [x] 3.5 Create `apps/train_ellipse_inf/README.md` with usage instructions

## 4. Validation

- [x] 4.1 Verify `packages/ellipse` can be imported and model instantiated
- [x] 4.2 Verify `train-ellipse` CLI command works and model uses package import
- [x] 4.3 Verify `train-ellipse-inf` CLI command works with raw images
- [x] 4.4 Verify model parameter count matches original implementation
- [x] 4.5 Run forward pass verification for both apps
