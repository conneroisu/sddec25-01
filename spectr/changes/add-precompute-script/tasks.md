## 1. Create Precompute Script

- [ ] 1.1 Create `training/precompute.py` with argparse CLI
- [ ] 1.2 Add Kaggle API integration to download OpenEDS datasets
- [ ] 1.3 Implement dataset combination (train + validation splits)
- [ ] 1.4 Add gamma correction preprocessing (gamma=0.8 LUT)
- [ ] 1.5 Add CLAHE preprocessing (clipLimit=1.5, tileGridSize=8x8)
- [ ] 1.6 Compute spatial_weights from segmentation labels
- [ ] 1.7 Compute dist_map (signed distance transform per class)
- [ ] 1.8 Add `preprocessed` column to indicate preprocessing status
- [ ] 1.9 Implement HuggingFace dataset push with resume capability
- [ ] 1.10 Add `--hf-repo`, `--output-dir`, `--no-push`, `--validate` CLI options

## 2. Update Training Scripts for Preprocessed Data

- [ ] 2.1 Add preprocessed dataset detection in `train_efficientvit_local.py` IrisDataset
- [ ] 2.2 Skip gamma/CLAHE when `preprocessed` column is True
- [ ] 2.3 Apply same changes to `train_efficientvit_tiny_local.py`
- [ ] 2.4 Apply same changes to `train_ellipse_local.py`
- [ ] 2.5 Apply same changes to Modal scripts (`train_efficientvit.py`, `train_ellipse.py`)
- [ ] 2.6 Apply same changes to Colab notebooks (`train_efficientvit.ipynb`, `train_ellipse.ipynb`)

## 3. Testing & Validation

- [ ] 3.1 Run precompute.py on small subset to verify preprocessing equivalence
- [ ] 3.2 Compare sample outputs: precomputed vs runtime preprocessing
- [ ] 3.3 Run training with preprocessed dataset and verify mIoU matches
- [ ] 3.4 Benchmark training speed improvement (epochs/second)

## 4. Documentation

- [ ] 4.1 Add usage instructions to `training/README.md`
- [ ] 4.2 Document Kaggle API credential setup
- [ ] 4.3 Update HuggingFace dataset card with preprocessing details
