# Change: Add Local Ellipse Training Script

## Why
The existing `train_ellipse.py` requires Modal cloud infrastructure to run, which adds deployment complexity and cost. A local version enables:
- Development and testing without Modal account/credits
- Training on local GPU workstations
- Easier debugging and experimentation
- CI/CD integration without cloud dependencies

## What Changes
- Add new file `training/train_ellipse_local.py` that mirrors `train_ellipse.py` functionality
- Remove Modal-specific code (Image, App, Volume, Secret, decorators)
- Replace Modal Volume caching with local filesystem caching
- Replace Modal secrets with environment variables or local config
- Add argparse for configurable hyperparameters
- Support both local disk and HuggingFace dataset loading

## Impact
- Affected specs: `training`
- Affected code: `training/` directory (new file only, no changes to existing)
- No breaking changes - existing Modal version remains functional
