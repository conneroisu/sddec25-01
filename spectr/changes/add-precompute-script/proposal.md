# Change: Add Dataset Precompute Script

## Why

Training runs are slowed by CPU-bound preprocessing performed per-sample during data loading. The current `IrisDataset.__getitem__` performs gamma correction (LUT), CLAHE, line augmentation, and Gaussian blur on every sample access. Moving static preprocessing (gamma + CLAHE) to a one-time precompute step will:

1. **Eliminate redundant computation**: CLAHE and gamma are applied identically every epoch
2. **Speed up training iteration**: Remove CPU bottleneck from dataloader
3. **Simplify training code**: Dataset class only needs to load precomputed tensors
4. **Enable GPU-native augmentation**: Keep only stochastic augmentations (flip, line, blur) in training loop

Currently the `Conner/openeds-precomputed` HuggingFace dataset contains raw images with precomputed `spatial_weights` and `dist_map`. This change adds a script to also precompute the deterministic image preprocessing.

## What Changes

- **ADDED** `training/precompute.py` - Downloads raw OpenEDS from Kaggle, combines train/validation splits, applies gamma correction and CLAHE, computes spatial weights and distance maps, and pushes to HuggingFace
- **MODIFIED** Training scripts - Update `IrisDataset` to skip gamma/CLAHE when loading from precomputed dataset
- **MODIFIED** HuggingFace dataset - New version with preprocessed images

## Impact

- Affected specs: `training`
- Affected code:
  - `training/precompute.py` (new)
  - `training/train_efficientvit_local.py`
  - `training/train_efficientvit_tiny_local.py`
  - `training/train_ellipse_local.py`
- External: HuggingFace dataset `Conner/openeds-precomputed` will be updated with v2
