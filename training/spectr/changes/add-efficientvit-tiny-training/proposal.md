# Change: Add Ultra-Tiny EfficientViT Training Script (<10k params)

## Why
The current `train_efficientvit_local.py` uses a model with ~57k parameters. For extreme edge deployment scenarios or research into minimal viable segmentation models, a version with less than 10,000 parameters is needed. This enables:
- Faster training iterations for hyperparameter search
- Deployment on ultra-constrained edge devices
- Research into minimal model capacity for eye segmentation
- Baseline comparison against larger models

## What Changes
- Add new file `training/train_efficientvit_tiny_local.py` based on `train_efficientvit_local.py`
- Reduce model dimensions: `embed_dims=(8, 12, 18)`, `decoder_dim=8`
- Reduce all attention heads to 1: `num_heads=(1, 1, 1)`
- Target parameter count: ~7,600 (well under 10k limit)
- Update MLflow tags to identify as "TinyEfficientViT-Micro" variant

## Impact
- Affected specs: `training`
- Affected code: `training/` directory (new file only)
- No breaking changes - existing scripts remain functional
