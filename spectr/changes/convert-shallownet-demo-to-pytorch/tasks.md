# Tasks: Convert ShallowNet Demo to PyTorch

## 1. Model Setup

- [ ] 1.1 Add ShallowNet model class to `demo/demo.py` (copy from `training/train.py`)
- [ ] 1.2 Add device detection function (CUDA > MPS > CPU)
- [ ] 1.3 Add `--device` CLI argument

## 2. Inference Conversion

- [ ] 2.1 Replace ONNX model loading with PyTorch state dict loading
- [ ] 2.2 Replace `_run_inference()` with PyTorch forward pass using `torch.no_grad()`
- [ ] 2.3 Update device display text ("CUDA"/"MPS"/"CPU")

## 3. Cleanup

- [ ] 3.1 Remove ONNX Runtime imports and IO binding code
- [ ] 3.2 Update `demo/requirements.txt` (remove onnxruntime, add torch)
- [ ] 3.3 Update README with new model format and MPS instructions

## 4. Validation

- [ ] 4.1 Test on MPS (Apple Silicon)
- [ ] 4.2 Test on CUDA (if available)
- [ ] 4.3 Test CPU fallback
