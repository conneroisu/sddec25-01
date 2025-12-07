## 1. Implementation

- [x] 1.1 Create `training/precompute.py` with CLI argument parsing (--hf-repo, --output-dir, --no-push, --validate, --skip-download)
- [x] 1.2 Implement Kaggle OpenEDS download using kagglehub (dataset: soumicksarker/openeds-dataset)
- [x] 1.3 Implement label binarization: pupil (class 3) → 1, all else → 0
- [x] 1.4 Implement empty mask filtering with logging of skipped samples
- [x] 1.5 Implement gamma correction preprocessing (LUT returns float64, convert to uint8 before CLAHE)
- [x] 1.6 Implement CPU CLAHE preprocessing (cv2.createCLAHE, clipLimit=1.5, tileGridSize=8x8)
- [x] 1.7 Implement ellipse parameter extraction (cv2.findContours + cv2.fitEllipse with moments fallback)
- [x] 1.8 Implement ellipse normalization (cx/640, cy/400, rx/377.36, ry/377.36)
- [x] 1.9 Implement spatial weights computation (morphological gradient + Gaussian blur sigma=5)
- [x] 1.10 Implement signed distance map computation (scipy EDT, normalized by diagonal)
- [x] 1.11 Implement HuggingFace dataset push with chunked parquet upload to Conner/sddec25-01
- [x] 1.12 Add validation mode to compare preprocessed vs runtime results

## 2. Training Script Updates

- [x] 2.1 Update `train_efficientvit.py` to detect `preprocessed` flag and skip gamma/CLAHE
- [x] 2.2 Update `train_efficientvit_local.py` to detect `preprocessed` flag and skip gamma/CLAHE
- [x] 2.3 Update `train_efficientvit_tiny_local.py` to detect `preprocessed` flag and skip gamma/CLAHE AND GPU CLAHE
- [x] 2.4 Update `train_ellipse.py` to use precomputed cx, cy, rx, ry instead of runtime extraction
- [x] 2.5 Update `train_ellipse_local.py` to use precomputed cx, cy, rx, ry instead of runtime extraction
- [x] 2.6 Update `train.py` (ShallowNet Modal) to detect `preprocessed` flag and skip gamma/CLAHE

## 3. Documentation

- [x] 3.1 Update `training/README.md` with precompute script usage and new HF repo (Conner/sddec25-01)
- [x] 3.2 Document Kaggle API credential setup (kaggle.json in ~/.kaggle/)
- [x] 3.3 Document HuggingFace dataset schema including ellipse parameters
- [x] 3.4 Document label binarization (4-class → 2-class)

## 4. Validation

- [ ] 4.1 Run precompute script on full OpenEDS dataset
- [ ] 4.2 Verify train/validation split sizes (expect ~27,400 train, ~2,775 validation minus empty masks)
- [ ] 4.3 Validate preprocessed images match runtime preprocessing (pixel-level comparison)
- [ ] 4.4 Validate ellipse parameters match runtime extraction
- [ ] 4.5 Validate spatial_weights shape is float32[400, 640]
- [ ] 4.6 Validate dist_map shape is float32[2, 400, 640]
- [ ] 4.7 Test training scripts with new preprocessed dataset (Conner/sddec25-01)
- [ ] 4.8 Log count of empty mask samples skipped
