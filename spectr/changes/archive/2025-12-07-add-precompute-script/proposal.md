# Change: Add Dataset Precompute Script

## Why

Training runs are slowed by CPU-bound preprocessing performed per-sample during data loading. The current `IrisDataset.__getitem__` performs gamma correction (LUT), CLAHE, and ellipse parameter extraction on every sample access. Moving ALL deterministic preprocessing to a one-time precompute step will:

1. **Eliminate redundant computation**: Gamma, CLAHE, and ellipse fitting are applied identically every epoch
2. **Speed up training iteration**: Remove CPU bottleneck from dataloader
3. **Simplify training code**: Dataset class only needs to load precomputed tensors
4. **Enable GPU-native augmentation**: Keep only stochastic augmentations (flip, line, blur) in training loop

This change adds a script that downloads the raw OpenEDS dataset from Kaggle (`soumicksarker/openeds-dataset`), applies all deterministic preprocessing, and uploads to HuggingFace repository `Conner/sddec25-01`.

## What Changes

- **ADDED** `training/precompute.py` - Downloads raw OpenEDS from Kaggle, applies gamma correction and CLAHE (CPU/OpenCV), computes ellipse parameters, spatial weights, distance maps, and pushes to HuggingFace
- **MODIFIED** Training scripts - Update `IrisDataset` to detect `preprocessed` flag and skip runtime gamma/CLAHE/ellipse extraction
- **ADDED** HuggingFace dataset `Conner/sddec25-01` - New preprocessed dataset matching OpenEDS structure

## Impact

- Affected specs: `training`
- Affected code:
  - `training/precompute.py` (new)
  - `training/train_efficientvit.py` (already has preprocessed detection)
  - `training/train_efficientvit_local.py` (already has preprocessed detection)
  - `training/train_efficientvit_tiny_local.py` (already has preprocessed detection + GPU CLAHE skip)
  - `training/train_ellipse.py` (needs precomputed ellipse params support)
  - `training/train_ellipse_local.py` (needs precomputed ellipse params support)
  - `training/train.py` (ShallowNet - needs preprocessed detection added)
  - `training/README.md` (update docs)
- External: New HuggingFace dataset `Conner/sddec25-01` with fully preprocessed images, binarized labels, and ellipse parameters
