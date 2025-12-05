# Implementation Tasks

## 1. Update Export Function
- [x] 1.1 Rename `export_model_to_onnx()` to `save_model_checkpoint()` in `train_ellipse.py:794`
- [x] 1.2 Replace ONNX export logic with PyTorch `state_dict()` saving
- [x] 1.3 Handle `torch.compile()` wrapped models (check for `_orig_mod` attribute)
- [x] 1.4 Convert model to contiguous format before saving for portability
- [x] 1.5 Update function docstring to reflect PyTorch checkpoint saving

## 2. Update Model Save Points
- [x] 2.1 Change best model filename from `best_ellipse_model.onnx` to `best_ellipse_model.pt` (line 1450)
- [x] 2.2 Change checkpoint filenames from `ellipse_model_epoch_{n}.onnx` to `ellipse_model_epoch_{n}.pt` (line 1483)
- [x] 2.3 Verify MLflow artifact logging still works with `.pt` files

## 3. Update Print Messages
- [x] 3.1 Update success message from "Model exported to ONNX" to "Model checkpoint saved"
- [x] 3.2 Update RuntimeError messages to reflect checkpoint saving (not ONNX export)

## 4. Validation
- [x] 4.1 Run `modal run training/train_ellipse.py` to verify script executes
- [x] 4.2 Confirm `.pt` files are generated (not `.onnx`)
- [x] 4.3 Verify MLflow logs the `.pt` artifacts correctly
- [x] 4.4 Test loading saved checkpoint with `torch.load()` and model instantiation
