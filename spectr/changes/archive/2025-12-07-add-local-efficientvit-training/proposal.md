# Change: Add Local EfficientViT Training Script

## Why
The existing `train_efficientvit.py` requires Modal cloud infrastructure to run, which adds deployment complexity and cost. A local version enables:
- Development and testing without Modal account/credits
- Training on local GPU workstations (e.g., with CUDA)
- Easier debugging and experimentation with hyperparameters
- CI/CD integration without cloud dependencies

## What Changes
- Add new file `training/train_efficientvit_local.py` that mirrors `train_efficientvit.py` functionality
- Remove Modal-specific code (Image, App, Volume, Secret, decorators)
- Replace Modal Volume caching with local filesystem caching (`~/.cache/openeds` or configurable)
- Replace Modal secrets with environment variables for MLflow credentials
- Add argparse for configurable hyperparameters (epochs, batch_size, lr, output_dir, etc.)
- Support both local disk and HuggingFace dataset loading
- Add device detection (CUDA/MPS/CPU) with user override option

## Impact
- Affected specs: `training`
- Affected code: `training/` directory (new file only, no changes to existing)
- No breaking changes - existing Modal version remains functional
