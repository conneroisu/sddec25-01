## 1. Implementation

- [x] 1.1 Copy `training/train_efficientvit_local.py` to `train_efficientvit_tiny_local.py` (preserves all GPU optimizations)
- [x] 1.2 Modify model configuration to use `embed_dims=(8, 12, 18)`, `decoder_dim=8`
- [x] 1.3 Reduce attention heads to `num_heads=(1, 1, 1)`
- [x] 1.4 Update checkpoint filenames to `best_efficientvit_tiny_model.pt` and `efficientvit_tiny_model_epoch_{n}.pt`
- [x] 1.5 Update parameter budget check from 60k to 10k
- [x] 1.6 Update MLflow tags to identify as "TinyEfficientViT-Micro" variant
- [x] 1.7 Update docstring, model_details, and run_name to reflect tiny variant

## 2. Validation

- [x] 2.1 Verify model instantiates and parameter count is under 10,000 (7,698 params)
- [x] 2.2 Verify forward pass produces correct output shape (B, 2, 400, 640)
- [x] 2.3 Confirm checkpoint filenames are distinct from standard model
- [x] 2.4 Verify GPU optimizations preserved (channels_last, torch.compile, autocast, GPU metrics)
