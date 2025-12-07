## Context

The VisionAssist training pipeline uses the OpenEDS dataset from Kaggle (`soumicksarker/openeds-dataset`) for eye pupil segmentation. Currently, deterministic preprocessing (gamma correction, CLAHE, ellipse parameter extraction) is applied at runtime during training, creating a CPU bottleneck.

**Constraints:**
- Must match OpenEDS Kaggle dataset structure (train/validation splits)
- Image dimensions: 640x400 grayscale
- Preprocessing must be identical to current training pipeline for consistency
- Training scripts must detect preprocessed data and skip redundant operations
- Use CPU CLAHE (OpenCV) as source of truth for consistency across all scripts
- Models use 2-class segmentation (num_classes=2)

## Goals / Non-Goals

**Goals:**
- Precompute ALL deterministic image preprocessing (gamma=0.8, CLAHE clipLimit=1.5)
- Binarize labels: pupil (class 3) -> 1, everything else (0,1,2) -> 0
- Precompute ellipse parameters (cx, cy, rx, ry) from binarized masks
- Precompute spatial weights and distance maps (2 channels for 2 classes)
- Upload to new HuggingFace repo `Conner/sddec25-01` with clear `preprocessed` flag
- Skip samples with empty pupil masks (no valid ellipse)
- Maintain backward compatibility with raw dataset loading

**Non-Goals:**
- Precomputing stochastic augmentations (RandomHorizontalFlip, Line_augment, Gaussian_blur)
- Modifying image dimensions or format
- Changing the underlying model architecture
- Using GPU CLAHE (Kornia) for precompute - CPU OpenCV is the source of truth

## Decisions

### Decision: Label Binarization
Convert OpenEDS 4-class labels to binary:
- Input: 0=background, 1=sclera, 2=iris, 3=pupil
- Output: 0=everything else, 1=pupil only

```python
binary_label = (raw_label == 3).astype(np.uint8)
```

**Rationale:** Models use `num_classes=2` and `F.one_hot(target, num_classes=2)`. Only pupil segmentation is needed.

### Decision: Preprocessing Pipeline Order
Apply preprocessing in this exact order:

```python
# 1. Create gamma LUT (float64)
gamma_table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)

# 2. Apply gamma correction via LUT (returns float64)
gamma_corrected = cv2.LUT(image, gamma_table)

# 3. Convert to uint8 (CRITICAL: must happen before CLAHE)
gamma_uint8 = np.uint8(gamma_corrected)

# 4. Apply CLAHE on uint8 image
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
preprocessed = clahe.apply(gamma_uint8)
```

**Rationale:** Matches exact training script behavior in `IrisDataset.__getitem__`:
- `cv2.LUT` returns float64 when given float64 LUT
- Training scripts convert to uint8 with `np.uint8(pilimg)` before CLAHE
- CLAHE expects and returns uint8

### Decision: Spatial Weights Computation (Industry Standard)
Compute boundary weights using morphological gradient:

```python
from scipy import ndimage

def compute_spatial_weights(label, sigma=5):
    # Morphological gradient to find boundaries
    dilated = ndimage.binary_dilation(label, iterations=1)
    eroded = ndimage.binary_erosion(label, iterations=1)
    boundary = dilated.astype(float) - eroded.astype(float)

    # Gaussian blur for smooth weight decay from boundary
    weights = ndimage.gaussian_filter(boundary, sigma=sigma)
    return weights.astype(np.float32)
```

**Rationale:** Industry standard approach - weights class boundaries higher in loss function to improve edge accuracy.

### Decision: Distance Map Computation (Industry Standard)
Compute signed distance transform per class using scipy EDT:

```python
from scipy.ndimage import distance_transform_edt

def compute_dist_map(label, num_classes=2):
    dist_map = np.zeros((num_classes, *label.shape), dtype=np.float32)
    for c in range(num_classes):
        class_mask = (label == c)
        if class_mask.any():
            # Distance inside class (negative)
            dist_inside = distance_transform_edt(class_mask)
            # Distance outside class (positive)
            dist_outside = distance_transform_edt(~class_mask)
            # Signed distance: negative inside, positive outside
            dist_map[c] = dist_outside - dist_inside
        else:
            # If class not present, all pixels are "far" from it
            dist_map[c] = distance_transform_edt(np.ones_like(label))

    # Normalize by max distance for stability
    max_dist = np.sqrt(label.shape[0]**2 + label.shape[1]**2)
    dist_map = dist_map / max_dist
    return dist_map
```

**Rationale:** Standard signed distance transform. Used in surface loss to penalize predictions far from true boundaries. Normalized for numerical stability.

### Decision: Ellipse Parameter Extraction
Precompute ellipse parameters using existing algorithm:
1. `cv2.findContours` on binary mask
2. Find largest contour
3. `cv2.fitEllipse` if contour has 5+ points
4. Fallback to moments-based circle if ellipse fit fails
5. Store normalized (cx, cy, rx, ry) values

Normalization constants:
- `IMAGE_WIDTH = 640`
- `IMAGE_HEIGHT = 400`
- `MAX_RADIUS = sqrt(640^2 + 400^2) / 2 = 377.36`

```python
cx_norm = cx / IMAGE_WIDTH      # 0-1 range
cy_norm = cy / IMAGE_HEIGHT     # 0-1 range
rx_norm = rx / MAX_RADIUS       # 0-1 range (typically)
ry_norm = ry / MAX_RADIUS       # 0-1 range (typically)
```

**Rationale:** Ellipse extraction is deterministic and CPU-intensive. Precomputing eliminates ~50ms per sample during training.

### Decision: Empty Mask Handling
Skip samples where the binarized pupil mask is empty (no pupil visible).

**Rationale:** User preference. These samples cannot provide valid ellipse parameters and would cause issues during ellipse training. Segmentation training also benefits from excluding these ambiguous samples.

### Decision: Dataset Schema
Store the following columns in the HuggingFace dataset:
- `image`: uint8[400, 640] - preprocessed grayscale image (gamma + CLAHE applied)
- `label`: uint8[400, 640] - binarized segmentation mask (0=background, 1=pupil)
- `spatial_weights`: float32[400, 640] - boundary weights for loss
- `dist_map`: float32[2, 400, 640] - signed distance per class (normalized)
- `cx`: float32 - normalized ellipse center x (0-1)
- `cy`: float32 - normalized ellipse center y (0-1)
- `rx`: float32 - normalized ellipse radius x
- `ry`: float32 - normalized ellipse radius y
- `filename`: string - original filename for traceability
- `preprocessed`: bool - always True for this dataset

**Rationale:** Extends existing schema with ellipse parameters for faster ellipse training.

### Decision: Kaggle Download
Use `kagglehub` with dataset ID `soumicksarker/openeds-dataset`.

**Rationale:** Official Kaggle source, programmatic download with caching.

### Decision: HuggingFace Repository
Push to new repository `Conner/sddec25-01`.

**Rationale:** Preserves existing `Conner/openeds-precomputed` for backward compatibility.

### Decision: Train/Validation Split
Use the same split as the original OpenEDS dataset:
- Training: subjects used for training in original dataset
- Validation: subjects used for validation in original dataset

The split is determined by the Kaggle dataset folder structure.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Large dataset size (~40GB) | Use chunked parquet upload to HuggingFace |
| Preprocessing mismatch | Validate by comparing runtime vs precomputed images |
| Kaggle auth issues | Document API credential setup clearly |
| Ellipse fit failures | Use moments-based fallback, log statistics |
| Empty mask samples | Skip and log count of excluded samples |
| Signed distance overflow | Normalize by image diagonal |

### Decision: GPU CLAHE Compatibility
`train_efficientvit_tiny_local.py` uses GPU CLAHE (Kornia) instead of CPU CLAHE:

```python
# GPU CLAHE in training loop (NOT in dataset)
data = (data + 0.5).clamp(0, 1)  # Denormalize from [-0.5, 0.5] to [0, 1]
data = K_enhance.equalize_clahe(data, clip_limit=1.5, grid_size=(8, 8))
data = data - 0.5  # Renormalize to [-0.5, 0.5]
```

When using preprocessed data, this GPU CLAHE step MUST be skipped (already handled in training script via `is_preprocessed_dataset` check).

**Note:** CPU CLAHE (OpenCV) and GPU CLAHE (Kornia) may have minor numerical differences due to:
- Different quantization (uint8 vs float)
- Different implementation details

The precompute script uses CPU CLAHE as the source of truth. Training scripts using GPU CLAHE will skip their CLAHE step when using preprocessed data.

## Stochastic Augmentations (Runtime Only)

These remain at training time:
- `RandomHorizontalFlip` (50% probability)
- `Line_augment` (20% probability) - random line overlay
- `Gaussian_blur` (20% probability) - cv2.GaussianBlur with random sigma

## Open Questions

None - all implementation details clarified.
