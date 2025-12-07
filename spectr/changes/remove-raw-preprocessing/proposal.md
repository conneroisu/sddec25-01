# Change: Remove Raw Preprocessing Code from Training Scripts

## Why

With the `add-precompute-script` change complete, all training now uses the preprocessed HuggingFace dataset (`Conner/sddec25-01`). The conditional logic for handling non-preprocessed data is now dead code that:

1. **Adds complexity**: if/else branches checking `preprocessed` flag
2. **Increases maintenance burden**: Two code paths to maintain
3. **Confuses readers**: Suggests raw data path is still supported
4. **Wastes lines**: ~50-100 lines per file of unused code

This change removes all raw preprocessing code paths, simplifying training scripts to assume `preprocessed=True` always.

## What Changes

- **REMOVED** `has_preprocessed_column` detection logic from all training scripts
- **REMOVED** `is_preprocessed` conditional checks
- **REMOVED** Runtime gamma correction code (`cv2.LUT`, `gamma_table`)
- **REMOVED** Runtime CLAHE code (`cv2.createCLAHE`, `clahe.apply()`)
- **REMOVED** Runtime ellipse extraction in ellipse training scripts
- **REMOVED** GPU CLAHE code in `train_efficientvit_tiny_local.py`
- **SIMPLIFIED** `IrisDataset.__getitem__` to load preprocessed data directly

## Impact

- Affected specs: `training`
- Affected code:
  - `training/train.py`
  - `training/train_efficientvit.py`
  - `training/train_efficientvit_local.py`
  - `training/train_efficientvit_tiny_local.py`
  - `training/train_ellipse.py`
  - `training/train_ellipse_local.py`
- **Breaking**: Training scripts will NO LONGER work with non-preprocessed datasets
