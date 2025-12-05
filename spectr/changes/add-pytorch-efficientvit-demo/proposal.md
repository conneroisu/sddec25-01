# Change: Add PyTorch-Based EfficientViT Demo with Multi-Device Support

## Why

The current `demo/demo.py` uses ONNX Runtime for inference, which provides good CPU/GPU support but requires model conversion. A native PyTorch demo enables:
1. Direct inference from trained `.pt` checkpoints without ONNX export
2. MPS (Metal Performance Shaders) support on Apple Silicon Macs for improved demo portability
3. Unified codebase where model definition is shared between training and inference
4. Easier debugging and model introspection during development

## What Changes

- Add new `demo/demo_pytorch.py` - PyTorch-native webcam demo for TinyEfficientViT
- Support three device backends: CUDA (NVIDIA GPUs), MPS (Apple Silicon), CPU
- Load trained model weights directly from `.pt` checkpoint files
- Reuse the exact TinyEfficientViT model definition from `training/train_efficientvit.py`
- Maintain identical preprocessing pipeline as training to ensure model accuracy
- Keep same visualization and user controls as existing ONNX demo

## Impact

- Affected specs: webcam-demo (MODIFIED - adds PyTorch inference mode)
- Affected code: `demo/demo_pytorch.py` (new file)
- Dependencies: PyTorch, torchvision, MediaPipe, OpenCV
- No changes to existing ONNX-based demo (both can coexist)
- Model compatibility: Uses same TinyEfficientViT architecture as training script

## Technical Considerations

### Device Selection Strategy
1. **Auto-detection (default)**: Check CUDA → MPS → CPU in order
2. **Explicit override**: `--device cuda|mps|cpu` argument
3. **Graceful fallback**: If requested device unavailable, fall back with warning

### Model Loading
- Load TinyEfficientViT model definition directly (no ONNX conversion)
- Support both `.pt` state dict and `.onnx` files (with conversion)
- Model expects input shape: (1, 1, H, W) grayscale with W=640, H=400

### Preprocessing Consistency
Critical to match training exactly (from `train_efficientvit.py`):
- Gamma correction: γ=0.8
- CLAHE: clipLimit=1.5, tileGridSize=(8,8)
- Resize to 640x400
- Normalize: mean=0.5, std=0.5
