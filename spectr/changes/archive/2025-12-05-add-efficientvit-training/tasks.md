## 1. Model Implementation

- [x] 1.1 Implement TinyConvNorm layer (conv + batchnorm, parameter-efficient)
- [x] 1.2 Implement TinyPatchEmbedding (2-layer, stride 4)
- [x] 1.3 Implement TinyCascadedGroupAttention (1-2 heads, small key_dim)
- [x] 1.4 Implement TinyLocalWindowAttention wrapper
- [x] 1.5 Implement TinyEfficientVitBlock (dw conv + attention + mlp)
- [x] 1.6 Implement TinyEfficientVitStage with optional downsampling
- [x] 1.7 Implement TinyEfficientVitEncoder (combines stages)
- [x] 1.8 Implement lightweight segmentation decoder with skip connections
- [x] 1.9 Implement TinyEfficientViTSeg (full model combining encoder + decoder)
- [x] 1.10 Add parameter counting and verify <60k total parameters

## 2. Training Script

- [x] 2.1 Create `train_efficientvit.py` with Modal app setup
- [x] 2.2 Copy dataset loading from train.py (IrisDataset, augmentations)
- [x] 2.3 Copy loss functions (CombinedLoss) and metrics (IoU computation)
- [x] 2.4 Copy visualization utilities (plots, predictions)
- [x] 2.5 Instantiate TinyEfficientViTSeg model with config
- [x] 2.6 Setup MLflow tracking with distinct experiment/run tags
- [x] 2.7 Implement training loop with AMP support
- [x] 2.8 Implement validation loop
- [x] 2.9 Add ONNX export for best model
- [x] 2.10 Add checkpoint saving at intervals

## 3. Validation

- [x] 3.1 Verify model parameter count is <60k (57,434 params with embed_dims=(16, 32, 64), decoder_dim=32)
- [x] 3.2 Verify model forward pass on 640x400 input produces 640x400 output
- [ ] 3.3 Verify ONNX export succeeds without errors (requires Modal runtime)
- [ ] 3.4 Test training script runs locally (CPU, 1 epoch, small batch)
- [ ] 3.5 Run full training on Modal and verify MLflow logging
