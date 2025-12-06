## 1. Implementation

- [ ] 1.1 Create `training/precompute.py` with CLI argument parsing (--hf-repo, --output-dir, --no-push, --validate, --skip-download)
- [ ] 1.2 Implement Kaggle OpenEDS download using kagglehub (dataset: soumicksarker/openeds-dataset)
- [ ] 1.3 Implement gamma correction preprocessing (gamma=0.8 LUT)
- [ ] 1.4 Implement CPU CLAHE preprocessing (cv2.createCLAHE, clipLimit=1.5, tileGridSize=8x8)
- [ ] 1.5 Implement ellipse parameter extraction (cv2.findContours + cv2.fitEllipse with moments fallback)
- [ ] 1.6 Implement spatial weights computation from segmentation labels
- [ ] 1.7 Implement signed distance map computation per class
- [ ] 1.8 Implement HuggingFace dataset push with chunked parquet upload to Conner/sddec25-01
- [ ] 1.9 Add validation mode to compare preprocessed vs runtime results

## 2. Training Script Updates

- [ ] 2.1 Update `train_efficientvit.py` to detect `preprocessed` flag and skip gamma/CLAHE
- [ ] 2.2 Update `train_efficientvit_local.py` to detect `preprocessed` flag and skip gamma/CLAHE
- [ ] 2.3 Update `train_efficientvit_tiny_local.py` to detect `preprocessed` flag and skip gamma/CLAHE (also skip GPU CLAHE)
- [ ] 2.4 Update `train_ellipse.py` to use precomputed cx, cy, rx, ry instead of runtime extraction
- [ ] 2.5 Update `train_ellipse_local.py` to use precomputed cx, cy, rx, ry instead of runtime extraction

## 3. Documentation

- [ ] 3.1 Update `training/README.md` with precompute script usage and new HF repo reference
- [ ] 3.2 Document Kaggle API credential setup (kaggle.json)
- [ ] 3.3 Document HuggingFace dataset schema including ellipse parameters

## 4. Validation

- [ ] 4.1 Run precompute script on full OpenEDS dataset
- [ ] 4.2 Verify train/validation split sizes match original (~27,400 train, ~2,775 validation)
- [ ] 4.3 Validate preprocessed images match runtime preprocessing (pixel-level comparison)
- [ ] 4.4 Validate ellipse parameters match runtime extraction
- [ ] 4.5 Test training scripts with new preprocessed dataset (Conner/sddec25-01)
