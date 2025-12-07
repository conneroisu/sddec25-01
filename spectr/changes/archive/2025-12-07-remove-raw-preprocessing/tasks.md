## 1. Code Removal

- [ ] 1.1 Remove `gamma_table` creation and `cv2.LUT` calls from `train.py`
- [ ] 1.2 Remove `clahe` creation and `clahe.apply()` calls from `train.py`
- [ ] 1.3 Remove `has_preprocessed_column` and `is_preprocessed` checks from `train.py`
- [ ] 1.4 Remove `gamma_table` creation and `cv2.LUT` calls from `train_efficientvit.py`
- [ ] 1.5 Remove `clahe` creation and `clahe.apply()` calls from `train_efficientvit.py`
- [ ] 1.6 Remove `has_preprocessed_column` and `is_preprocessed` checks from `train_efficientvit.py`
- [ ] 1.7 Remove `gamma_table` creation and `cv2.LUT` calls from `train_efficientvit_local.py`
- [ ] 1.8 Remove `clahe` creation and `clahe.apply()` calls from `train_efficientvit_local.py`
- [ ] 1.9 Remove `has_preprocessed_column` and `is_preprocessed` checks from `train_efficientvit_local.py`
- [ ] 1.10 Remove `gamma_table` creation and `cv2.LUT` calls from `train_efficientvit_tiny_local.py`
- [ ] 1.11 Remove GPU CLAHE (`K_enhance.equalize_clahe`) from `train_efficientvit_tiny_local.py`
- [ ] 1.12 Remove `has_preprocessed_column` and `is_preprocessed` checks from `train_efficientvit_tiny_local.py`
- [ ] 1.13 Remove `gamma_table` creation and `cv2.LUT` calls from `train_ellipse.py`
- [ ] 1.14 Remove `clahe` creation and `clahe.apply()` calls from `train_ellipse.py`
- [ ] 1.15 Remove `has_preprocessed_column` and `is_preprocessed` checks from `train_ellipse.py`
- [ ] 1.16 Remove runtime `extract_ellipse_params` call from `train_ellipse.py` (use precomputed)
- [ ] 1.17 Remove `gamma_table` creation and `cv2.LUT` calls from `train_ellipse_local.py`
- [ ] 1.18 Remove `clahe` creation and `clahe.apply()` calls from `train_ellipse_local.py`
- [ ] 1.19 Remove `has_preprocessed_column` and `is_preprocessed` checks from `train_ellipse_local.py`
- [ ] 1.20 Remove runtime `extract_ellipse_params` call from `train_ellipse_local.py` (use precomputed)

## 2. Simplification

- [ ] 2.1 Simplify `IrisDataset.__init__` in all scripts (remove clahe/gamma_table init)
- [ ] 2.2 Simplify `IrisDataset.__getitem__` to directly use preprocessed image
- [ ] 2.3 Update ellipse scripts to load cx, cy, rx, ry from dataset columns

## 3. Cleanup

- [ ] 3.1 Remove unused imports (cv2.createCLAHE related if applicable)
- [ ] 3.2 Remove kornia import from `train_efficientvit_tiny_local.py` if no longer needed
- [ ] 3.3 Update comments/docstrings to reflect preprocessed-only workflow

## 4. Validation

- [ ] 4.1 Test `train_efficientvit_local.py` with preprocessed dataset
- [ ] 4.2 Test `train_ellipse_local.py` with preprocessed dataset
- [ ] 4.3 Verify training metrics are unchanged
