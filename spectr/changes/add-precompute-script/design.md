# Design: Dataset Precompute Script

## Context

The VisionAssist training pipeline uses the OpenEDS dataset from Kaggle. Currently, preprocessing happens at two stages:

1. **One-time precompute** (already done): `spatial_weights` and `dist_map` are computed from segmentation labels and stored in HuggingFace
2. **Per-sample runtime** (current bottleneck): gamma correction and CLAHE are applied every time a sample is loaded

This wastes GPU cycles waiting for CPU preprocessing. The bottleneck is especially pronounced with:
- Large batch sizes (64+)
- Multi-epoch training (25+ epochs)
- Fast GPUs (A100, H100) that can outpace CPU preprocessing

## Goals / Non-Goals

**Goals:**
- Create `precompute.py` that downloads raw Kaggle data, applies all deterministic preprocessing, and pushes to HuggingFace
- Maintain exact preprocessing semantics (gamma=0.8, CLAHE clipLimit=1.5, tileGridSize=8x8)
- Store preprocessed images in uint8 format to minimize dataset size
- Support incremental updates (add new splits without reprocessing existing)

**Non-Goals:**
- Runtime augmentations (Line_augment, Gaussian_blur, RandomHorizontalFlip) remain in training loop since they're stochastic
- Changing the actual preprocessing parameters (same gamma, CLAHE settings)
- GPU-accelerated precomputation (Python script is run once, speed not critical)

## Decisions

### Decision: Store preprocessed images as uint8

**Rationale**: CLAHE output is 8-bit grayscale. Storing as float32 would 4x dataset size with no accuracy benefit. ToTensor() normalization happens after loading.

### Decision: Keep stochastic augmentations in training loop

**Rationale**: Line_augment, Gaussian_blur, and RandomHorizontalFlip must vary per epoch. Moving them to precompute would reduce augmentation diversity.

**Alternatives considered**:
1. Pre-generate N augmented versions per sample - Increases storage by Nx, reduces diversity
2. GPU-based runtime augmentation with Kornia - Already implemented in `train_efficientvit_tiny_local.py`, orthogonal to this change

### Decision: Use Kaggle API for raw data download

**Rationale**: OpenEDS is hosted on Kaggle. The Kaggle Python API provides authenticated download with progress tracking.

### Decision: Add `--preprocessed` flag to training scripts

**Rationale**: Backward compatibility. Training scripts can detect dataset version and skip CLAHE/gamma if already preprocessed. Default behavior checks for `image_preprocessed` column.

## Data Pipeline

```
Kaggle OpenEDS (raw)
    │
    ▼
precompute.py
    ├── Download train/validation splits
    ├── Gamma correction (LUT, gamma=0.8)
    ├── CLAHE (clipLimit=1.5, tileGridSize=8x8)
    ├── Compute spatial_weights (edge distance)
    ├── Compute dist_map (signed distance transform)
    └── Push to HuggingFace
    │
    ▼
Conner/openeds-precomputed v2
    │
    ▼
IrisDataset.__getitem__
    ├── Load preprocessed image (uint8)
    ├── Apply stochastic augmentations (train only)
    ├── ToTensor + Normalize
    └── Return batch
```

## HuggingFace Dataset Schema

Current schema:
- `image`: uint8[400, 640] - raw grayscale
- `label`: uint8[400, 640] - segmentation mask
- `spatial_weights`: float32[400, 640] - edge weights
- `dist_map`: float32[2, 400, 640] - signed distance per class
- `filename`: string

New schema (v2):
- `image`: uint8[400, 640] - **preprocessed** (gamma + CLAHE applied)
- `label`: uint8[400, 640] - segmentation mask (unchanged)
- `spatial_weights`: float32[400, 640] - edge weights (unchanged)
- `dist_map`: float32[2, 400, 640] - signed distance per class (unchanged)
- `filename`: string (unchanged)
- `preprocessed`: bool - True for v2 samples (enables backward compat detection)

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| HuggingFace dataset push fails mid-upload | Use chunked upload, add resume capability |
| Preprocessing differs from training code | Extract shared preprocessing module, test equivalence |
| Kaggle rate limits | Add retry with exponential backoff |
| Dataset size increases | Store as uint8, use parquet compression |

## Open Questions

1. Should we version the HuggingFace dataset (v1, v2) or overwrite?
   - **Proposal**: Add `preprocessed` column, overwrite repo. Old code can check column and apply preprocessing if missing.

2. Should precompute.py be Modal-based for parallelism?
   - **Proposal**: No, keep simple Python script. Precompute runs once, parallelism not needed.
