# Tasks: Add Ultra-Tiny EfficientViT Training Script

## 1. Implementation

- [ ] 1.1 Copy `train_efficientvit_local.py` to `train_efficientvit_tiny_local.py`
- [ ] 1.2 Update model configuration to ultra-tiny dimensions:
  - `embed_dims=(8, 12, 18)` (down from `(16, 32, 64)`)
  - `decoder_dim=8` (down from `32`)
  - `num_heads=(1, 1, 1)` (down from `(1, 1, 2)`)
- [ ] 1.3 Update parameter budget check from 60k to 10k
- [ ] 1.4 Update MLflow tags to identify as "TinyEfficientViT-Micro"
- [ ] 1.5 Update checkpoint filenames to use `_tiny_` prefix
- [ ] 1.6 Update docstring and help text to reflect tiny variant

## 2. Validation

- [ ] 2.1 Verify model instantiates correctly with new config
- [ ] 2.2 Verify parameter count is under 10,000
- [ ] 2.3 Verify forward pass produces correct output shape (B, 2, 400, 640)
- [ ] 2.4 Run short training test (1-2 epochs) to verify training loop works
