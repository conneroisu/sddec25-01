## 1. Model Integration

- [x] 1.1 Extract TinyEfficientViT model classes to shared module `training/models/efficientvit.py`
- [x] 1.2 Update `train_efficientvit.py` to import from shared module
- [x] 1.3 Add PyTorch checkpoint saving (`.pt` state dict) alongside ONNX export
- [x] 1.4 Verify model loads correctly from saved `.pt` checkpoint

## 2. Demo Implementation

- [x] 2.1 Create `demo/demo_pytorch.py` with basic structure (argument parsing, main loop)
- [x] 2.2 Implement device detection (CUDA → MPS → CPU) with explicit override
- [x] 2.3 Implement model loading from `.pt` checkpoint
- [x] 2.4 Implement preprocessing pipeline matching training exactly
- [x] 2.5 Implement PyTorch inference with `torch.no_grad()`
- [x] 2.6 Implement post-processing (argmax, mask extraction)
- [x] 2.7 Integrate webcam capture (reuse from existing demo)
- [x] 2.8 Integrate MediaPipe face mesh (reuse from existing demo)
- [x] 2.9 Implement visualization overlay (reuse from existing demo)
- [x] 2.10 Add performance metrics display (FPS, inference time, device)

## 3. Device Support Validation

- [x] 3.1 Test on CUDA device (NVIDIA GPU)
- [x] 3.2 Test on MPS device (Apple Silicon Mac)
- [x] 3.3 Test on CPU-only system
- [x] 3.4 Verify graceful fallback when requested device unavailable

## 4. Documentation

- [x] 4.1 Update `demo/README.md` with PyTorch demo usage instructions
- [x] 4.2 Document device selection behavior
- [x] 4.3 Document model checkpoint requirements
- [x] 4.4 Add troubleshooting for common issues (MPS limitations, CUDA not found)
