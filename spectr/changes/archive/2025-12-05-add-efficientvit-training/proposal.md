# Change: Add EfficientViT Training Script for Eye Segmentation

## Why

The project currently trains only a ShallowNet U-Net model for pupil segmentation. Adding an EfficientViT-based model provides an alternative architecture that leverages attention mechanisms for potentially better boundary detection, while maintaining the strict <60k parameter constraint required for edge deployment on the Kria KV260.

## What Changes

- Add new `train_efficientvit.py` script in `training/` directory
- Implement a custom **TinyEfficientViT** segmentation model (<60k parameters)
- Reuse existing dataset loading, loss functions, augmentation, and MLflow logging from `train.py`
- Model outputs dense predictions at 640x400 resolution (same as ShallowNet)

## Impact

- Affected specs: training (new capability)
- Affected code: `training/train_efficientvit.py` (new file)
- No breaking changes to existing ShallowNet training pipeline
- Model size constraint: <60,000 trainable parameters (approximately 240KB at FP32)

## Technical Considerations

The standard EfficientViT-MSRA models (efficientvit_m0-m5) have hundreds of thousands to millions of parameters, far exceeding the 60k limit. A custom **TinyEfficientViT** must be designed with:

1. **Drastically reduced embedding dimensions**: embed_dim ~(8, 16, 24) instead of (64, 128, 192)
2. **Minimal attention heads**: 1-2 heads instead of 4
3. **Shallow depth**: depth=(1, 1, 1) instead of (1, 2, 3)
4. **Lightweight decoder**: Simple upsampling with minimal convolutions
5. **Smaller patch embedding**: Fewer convolution stages or smaller channels

Parameter budget breakdown (target ~55k):
- Patch embedding: ~3k params
- Stage 1 (8 dim, depth 1): ~5k params
- Stage 2 (16 dim, depth 1): ~10k params
- Stage 3 (24 dim, depth 1): ~15k params
- Decoder (upsampling path): ~20k params
- Classification head: ~2k params
