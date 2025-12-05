# Design: PyTorch EfficientViT Demo

## Context

The existing demo (`demo/demo.py`) uses ONNX Runtime for inference. This design extends the webcam demo capability to support native PyTorch inference with the TinyEfficientViT model trained in `training/train_efficientvit.py`.

**Stakeholders**: Developers demoing on various hardware (NVIDIA GPUs, Apple Silicon, CPU-only laptops)

**Constraints**:
- Must maintain identical preprocessing to training for model accuracy
- Must support real-time inference (~30 FPS on GPU, acceptable degradation on CPU)
- Model architecture must exactly match training definition

## Goals / Non-Goals

### Goals
- Native PyTorch inference with CUDA, MPS, and CPU support
- Direct loading of `.pt` checkpoint files from training
- Same user experience as ONNX demo (controls, visualization, metrics)
- Code reuse of model definition from training script

### Non-Goals
- Replacing ONNX demo (both will coexist)
- Supporting other model architectures (ShallowNet uses ONNX)
- Model quantization or optimization (out of scope)

## Decisions

### Decision 1: Model Definition Reuse
**What**: Import TinyEfficientViT model classes directly from training script
**Why**:
- Ensures architecture parity between training and inference
- Single source of truth for model definition
- Avoids subtle bugs from re-implementing model

**Alternatives considered**:
- Duplicate model code in demo (rejected: maintenance burden, drift risk)
- Create shared module (rejected: over-engineering for single model)

### Decision 2: Device Selection Priority
**What**: Auto-detect in order: CUDA → MPS → CPU, with explicit override
**Why**:
- CUDA most performant for NVIDIA users
- MPS enables Apple Silicon demos without manual config
- CPU fallback ensures demo always works

**Implementation**:
```python
def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

### Decision 3: Model Loading Strategy
**What**: Load state dict from `.pt` file into freshly instantiated model
**Why**:
- `.pt` files are what training produces (no conversion needed)
- Explicit model instantiation ensures correct architecture
- Clear error messages if shapes mismatch

**Model instantiation** (must match training config):
```python
model = TinyEfficientViTSeg(
    in_channels=1,
    num_classes=2,
    embed_dims=(16, 32, 64),
    depths=(1, 1, 1),
    num_heads=(1, 1, 2),
    key_dims=(4, 4, 4),
    attn_ratios=(2, 2, 2),
    window_sizes=(7, 7, 7),
    mlp_ratios=(2, 2, 2),
    decoder_dim=32,
)
```

### Decision 4: Preprocessing Pipeline
**What**: Exact match to training preprocessing in correct order
**Why**: Model accuracy depends on identical preprocessing

**Pipeline** (order matters):
1. Resize eye crop to 640x400 (INTER_LINEAR)
2. Convert to grayscale
3. Gamma correction (γ=0.8) via LUT
4. CLAHE (clipLimit=1.5, tileGridSize=8x8)
5. Normalize (mean=0.5, std=0.5)
6. Transpose to (W, H) for model input convention
7. Add batch/channel dims: (1, 1, W, H)

### Decision 5: Inference Mode
**What**: Use `torch.no_grad()` and `model.eval()` for inference
**Why**: Disables gradient computation and batch norm training behavior

**Additional optimizations**:
- Pre-allocate input tensor on target device
- Use `torch.inference_mode()` on PyTorch 1.9+ for better performance

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| MPS backend has bugs/limitations | Test thoroughly; fallback to CPU if issues |
| Model architecture drift between training and demo | Import from single source (training script) |
| Performance regression vs ONNX | Benchmark both; ONNX may be faster on some hardware |
| Memory usage on CPU | Monitor and document requirements |

## Migration Plan

1. Create new `demo/demo_pytorch.py` alongside existing `demo/demo.py`
2. Both demos coexist - users choose based on model format
3. Document when to use each:
   - `.pt` checkpoint → use `demo_pytorch.py`
   - `.onnx` model → use `demo.py`
4. No breaking changes to existing workflow

## Open Questions

1. Should we support loading ONNX models and converting to PyTorch? (Probably not - adds complexity)
2. Should we benchmark PyTorch vs ONNX performance? (Nice to have for documentation)
