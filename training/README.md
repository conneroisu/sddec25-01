# ShallowNet Training

Training script for the ShallowNet semantic segmentation model used in the VisionAssist project.

## Usage

```bash
uv run train.py
```

This runs the training on Modal with GPU acceleration.

## Dataset Precomputation

The `precompute.py` script preprocesses the OpenEDS dataset and pushes it to HuggingFace. This eliminates CPU-bound preprocessing during training for faster iteration.

### Prerequisites

Set up Kaggle API credentials:
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token" to download `kaggle.json`
3. Place `kaggle.json` in `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Usage

**Full preprocessing and push to HuggingFace:**
```bash
python precompute.py --hf-repo Conner/openeds-precomputed
```

**Local preprocessing only (no HuggingFace push):**
```bash
python precompute.py --no-push --output-dir ./my_dataset
```

**Validate existing dataset:**
```bash
python precompute.py --validate --hf-repo Conner/openeds-precomputed
```

**Skip Kaggle download (use existing files):**
```bash
python precompute.py --skip-download --output-dir ./existing_data
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--hf-repo` | `Conner/openeds-precomputed` | HuggingFace repo to push to |
| `--output-dir` | `./precompute_output` | Local directory for intermediate files |
| `--no-push` | - | Skip pushing to HuggingFace |
| `--validate` | - | Validate existing dataset preprocessing |
| `--skip-download` | - | Skip Kaggle download (use existing files) |

### Preprocessing Pipeline

1. **Download**: Fetches OpenEDS train/validation from Kaggle
2. **Gamma correction**: Applies LUT with gamma=0.8
3. **CLAHE**: Adaptive histogram equalization (clipLimit=1.5, tileGridSize=8x8)
4. **Spatial weights**: Computes boundary weights from segmentation labels
5. **Distance maps**: Signed distance transform per class
6. **Push**: Uploads to HuggingFace with chunked upload

### Output Schema

The preprocessed HuggingFace dataset contains:
- `image`: uint8[400, 640] - preprocessed grayscale
- `label`: uint8[400, 640] - segmentation mask
- `spatial_weights`: float32[400, 640] - boundary weights
- `dist_map`: float32[2, 400, 640] - signed distance per class
- `filename`: string
- `preprocessed`: bool (True for preprocessed samples)

Training scripts automatically detect the `preprocessed` flag and skip runtime preprocessing.
