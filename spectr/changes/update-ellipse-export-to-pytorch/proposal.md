# Change: Update Ellipse Training to Export PyTorch Models

## Why

The `train_ellipse.py` script currently exports models in ONNX format (`.onnx`), which is inconsistent with the newer `train_efficientvit.py` script that exports native PyTorch checkpoints (`.pt`). PyTorch checkpoints provide better compatibility with PyTorch workflows, easier model loading for fine-tuning, and preserve the full model state without conversion overhead.

## What Changes

- Replace ONNX export function with PyTorch checkpoint saving in `training/train_ellipse.py`
- Update model checkpoint filenames from `.onnx` to `.pt` extension
- Modify the export logic to save `state_dict()` instead of using `torch.onnx.export()`
- Maintain MLflow artifact logging for PyTorch checkpoints
- Align ellipse training export behavior with EfficientViT training patterns

## Impact

- **Affected specs**: `training` (modifies model export requirements)
- **Affected code**: `training/train_ellipse.py` (lines 794-822, 1450-1451, 1483-1484)
- **Compatibility**: Changes output artifact format from `.onnx` to `.pt` files
- **Migration**: Existing MLflow runs with ONNX artifacts remain valid; new runs will use PyTorch format
