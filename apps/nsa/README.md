# NSA - Native Sparse Attention for Pupil Segmentation

Implementation of Native Sparse Attention (NSA) adapted for pupil segmentation in eye images.

## Overview

This module implements the NSA mechanism from DeepSeek's paper "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" adapted for 2D vision/segmentation tasks.

### Key Components

1. **Token Compression**: Aggregates spatial blocks into coarse-grained tokens for global context
2. **Token Selection**: Selects important spatial regions based on attention importance scores
3. **Sliding Window**: Local attention for precise boundary delineation
4. **Gated Aggregation**: Learned combination of all attention paths

### Domain Adaptations for Pupil Segmentation

- **Intense Pixel Localization**: Selection attention focuses on the pupil region
- **Spatial Locality**: The pupil is only found on the eye, enabling efficient sparse selection
- **Multi-scale Processing**: Encoder-decoder architecture with skip connections

## Installation

```bash
cd nsa
uv sync
```

## Usage

### Training

```bash
uv run python train.py --model-size small --epochs 30
```

### Demo

```bash
uv run python demo.py
```

### Testing

```bash
uv run python test.py
```

## Model Configurations

| Size   | Parameters | Embed Dims     | Depths  |
|--------|------------|----------------|---------|
| tiny   | ~50K       | (16, 32, 48)   | (1,1,1) |
| small  | ~150K      | (32, 64, 96)   | (1,1,1) |
| medium | ~400K      | (48, 96, 144)  | (2,2,2) |

## Architecture

```
Input Image (1, H, W)
       │
       ▼
┌─────────────────┐
│  Patch Embed    │  4x downsample
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NSA Stage 1    │  Compress + Select + Window
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NSA Stage 2    │  2x downsample + NSA
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NSA Stage 3    │  2x downsample + NSA
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FPN Decoder    │  Multi-scale fusion
└────────┬────────┘
         │
         ▼
   Output (2, H, W)
```

## References

- [Native Sparse Attention Paper](https://arxiv.org/abs/2502.11089)
- [OpenEDS Dataset](https://research.facebook.com/publications/openeds-2020-challenge-on-eye-tracking-for-vr-and-ar/)
