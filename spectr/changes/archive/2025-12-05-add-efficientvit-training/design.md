## Context

The project needs an alternative neural network architecture for eye pupil segmentation that:
- Uses attention mechanisms (Vision Transformer style)
- Maintains strict <60k parameter constraint for edge deployment
- Outputs dense segmentation maps at 640x400 resolution
- Can be trained on the same OpenEDS dataset as ShallowNet

The standard EfficientViT-MSRA architectures are too large (300k+ parameters for even the smallest variant).

## Goals / Non-Goals

### Goals
- Create a TinyEfficientViT model with <60k trainable parameters
- Achieve comparable or better mIoU to ShallowNet (~99% pupil IoU target)
- Export to ONNX format for edge deployment
- Reuse existing training infrastructure (dataset, loss, logging)

### Non-Goals
- Not replacing ShallowNet (this is an alternative model)
- Not optimizing for inference speed (parameter count is primary constraint)
- Not supporting multi-class segmentation beyond background/pupil

## Decisions

### Decision 1: Model Architecture - TinyEfficientViT for Segmentation

We adapt EfficientViT-MSRA to segmentation with drastically reduced dimensions:

```python
# TinyEfficientViT Configuration (<60k params)
embed_dim = (8, 16, 24)      # vs original (64, 128, 192)
key_dim = (4, 4, 4)          # vs original (16, 16, 16)
depth = (1, 1, 1)            # vs original (1, 2, 3)
num_heads = (1, 1, 2)        # vs original (4, 4, 4)
window_size = (7, 7, 7)      # keep same for attention patterns
kernels = (3, 3, 3, 3)       # smaller kernels
```

**Why**: These dimensions achieve ~55k parameters while maintaining the core EfficientViT architecture (cascaded group attention, local window attention, patch merging).

### Decision 2: Segmentation Decoder

Use a lightweight FPN-style decoder with skip connections:

```python
class TinySegmentationDecoder(nn.Module):
    # Takes multi-scale features from stages [0, 1, 2]
    # Progressive upsampling with lateral connections
    # Final 1x1 conv for 2-class output
```

**Why**: Skip connections from encoder stages help preserve spatial detail for dense prediction without adding many parameters.

### Decision 3: Patch Embedding Simplification

Reduce patch embedding from 4 conv layers to 2:

```python
# Original: 4 convs, stride 16 total
# Ours: 2 convs, stride 4 total (preserve more spatial info)
conv1: in_chans -> dim//2, stride 2
conv2: dim//2 -> dim, stride 2
# Total stride: 4 (not 16)
```

**Why**: For segmentation, we need more spatial resolution. Smaller stride means more feature map pixels, but with tiny embed_dim this is still efficient.

### Decision 4: Input Resolution Handling

Keep original 640x400 resolution (no resizing):
- Matches ShallowNet training exactly
- Window attention handles non-square images via padding
- Output shape matches input shape

**Alternatives considered**:
- Resize to 224x224: Would lose detail, not directly comparable to ShallowNet
- Resize to 256x160: Maintains aspect ratio but still loses resolution

### Decision 5: Training Configuration

Inherit from train.py:
- Same optimizer (Adam), scheduler (ReduceLROnPlateau)
- Same loss function (CombinedLoss: CE + Dice + Surface)
- Same augmentations (flip, line, blur, CLAHE)
- Same MLflow experiment tracking

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Attention at low dim may not learn useful patterns | Use residual connections, careful initialization |
| May underperform ShallowNet at same param count | Accept tradeoff - this explores different architecture |
| Window attention expensive at 640x400 | Small window (7x7), few heads (1-2) |
| ONNX export complexity | Test early, use simple ops |

## Migration Plan

No migration needed - this adds a new training script without modifying existing code.

1. Implement `train_efficientvit.py` with TinyEfficientViT model
2. Run training on Modal (same infrastructure as train.py)
3. Compare results with ShallowNet baseline
4. Export best model to ONNX

## Open Questions

1. **Parameter allocation**: Should more parameters go to encoder or decoder?
   - Initial approach: 35k encoder, 20k decoder
   - May need tuning based on results

2. **Attention pattern**: Is cascaded group attention beneficial at 1-2 heads?
   - Alternative: Simple self-attention without grouping
   - Will evaluate empirically
