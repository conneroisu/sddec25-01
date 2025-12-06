## 1. Implementation

- [ ] 1.1 Create `training/train_efficientvit_tiny_local.py` based on `train_efficientvit_local.py`
- [ ] 1.2 Modify model configuration to use `embed_dims=(8, 12, 18)`, `decoder_dim=8`
- [ ] 1.3 Reduce attention heads to `num_heads=(1, 1, 1)`
- [ ] 1.4 Update MLflow tags to identify as "TinyEfficientViT-Micro" variant
- [ ] 1.5 Verify total parameter count is under 10,000

## 2. Testing

- [ ] 2.1 Run training script and verify it completes without errors
- [ ] 2.2 Verify model exports to ONNX format
- [ ] 2.3 Confirm MLflow logs are correctly tagged
