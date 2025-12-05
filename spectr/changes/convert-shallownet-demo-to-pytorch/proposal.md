# Change: Convert ShallowNet Demo from ONNX Runtime to PyTorch

## Why

ONNX Runtime lacks Apple Silicon MPS support. PyTorch has supported MPS since 2022, enabling GPU-accelerated inference on M1/M2/M3 Macs. This provides 2-5x speedup over CPU on Apple Silicon for demo presentations.

Reference: GitHub Issue #146

## What Changes

- **MODIFIED** `demo/demo.py` - Replace ONNX Runtime with PyTorch inference
- **ADDED** ShallowNet model class - Copy architecture from `training/train.py`
- **ADDED** Multi-device support - Auto-detect CUDA > MPS > CPU with `--device` override
- **MODIFIED** Model loading - Accept `.pt` state dict files
- **REMOVED** ONNX Runtime dependency

## Impact

- Affected specs: `webcam-demo` (MODIFIED - inference backend)
- Affected code: `demo/demo.py`
- **BREAKING**: Requires `.pt` model files instead of `.onnx`
- Dependencies: Remove `onnxruntime`, add `torch`
