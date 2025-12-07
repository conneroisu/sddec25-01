## ADDED Requirements

### Requirement: Dataset Precompute Script
The system SHALL provide a script `training/precompute.py` that downloads the OpenEDS dataset from Kaggle, applies all deterministic preprocessing, and uploads to HuggingFace.

#### Scenario: Full preprocessing pipeline
- **WHEN** user runs `python training/precompute.py --hf-repo Conner/sddec25-01`
- **THEN** script SHALL download OpenEDS train and validation splits from Kaggle (soumicksarker/openeds-dataset)
- **AND** binarize labels: pupil (class 3) -> 1, everything else -> 0
- **AND** skip samples with empty pupil masks
- **AND** apply gamma correction (gamma=0.8 via LUT) to all images
- **AND** apply CPU CLAHE (cv2.createCLAHE, clipLimit=1.5, tileGridSize=8x8) to all images
- **AND** extract ellipse parameters (cx, cy, rx, ry) from binarized masks via cv2.fitEllipse
- **AND** compute spatial weights via morphological gradient with Gaussian smoothing
- **AND** compute signed distance maps per class using scipy.ndimage.distance_transform_edt
- **AND** upload to HuggingFace repository Conner/sddec25-01

#### Scenario: Local-only preprocessing
- **WHEN** user runs `python training/precompute.py --no-push --output-dir ./local`
- **THEN** script SHALL preprocess all images locally
- **AND** save to specified output directory
- **AND** NOT push to HuggingFace

#### Scenario: Validation mode
- **WHEN** user runs `python training/precompute.py --validate --hf-repo Conner/sddec25-01`
- **THEN** script SHALL download samples from HuggingFace
- **AND** compare preprocessed images against runtime preprocessing
- **AND** compare ellipse parameters against runtime extraction
- **AND** report any mismatches

#### Scenario: Ellipse extraction fallback
- **WHEN** segmentation mask contour has fewer than 5 points
- **THEN** script SHALL use moments-based circle approximation
- **AND** set rx = ry = sqrt(area / pi)

#### Scenario: Empty mask handling
- **WHEN** binarized pupil mask is empty (no pupil pixels)
- **THEN** script SHALL skip the sample entirely
- **AND** log the skipped filename for traceability

### Requirement: Label Binarization
The precompute script SHALL convert OpenEDS 4-class labels to binary format for 2-class segmentation.

#### Scenario: Pupil extraction
- **WHEN** raw label contains classes 0 (background), 1 (sclera), 2 (iris), 3 (pupil)
- **THEN** output label SHALL contain only 0 (background) and 1 (pupil)
- **AND** class 3 pixels SHALL become 1
- **AND** classes 0, 1, 2 pixels SHALL become 0

### Requirement: Spatial Weights Computation
The precompute script SHALL compute boundary weights using morphological gradient.

#### Scenario: Boundary detection
- **WHEN** computing spatial weights for a label mask
- **THEN** script SHALL compute morphological gradient (dilation - erosion)
- **AND** apply Gaussian smoothing (sigma=5) for weight decay
- **AND** output shape SHALL be float32[400, 640]

### Requirement: Distance Map Computation
The precompute script SHALL compute signed distance transform per class.

#### Scenario: Signed distance transform
- **WHEN** computing distance map for 2-class segmentation
- **THEN** script SHALL use scipy.ndimage.distance_transform_edt
- **AND** compute negative distance inside each class
- **AND** compute positive distance outside each class
- **AND** normalize by image diagonal (sqrt(640^2 + 400^2))
- **AND** output shape SHALL be float32[2, 400, 640]

### Requirement: Ellipse Parameter Normalization
The precompute script SHALL normalize ellipse parameters for consistent training.

#### Scenario: Normalization scheme
- **WHEN** ellipse parameters are extracted
- **THEN** cx SHALL be normalized by IMAGE_WIDTH (640)
- **AND** cy SHALL be normalized by IMAGE_HEIGHT (400)
- **AND** rx and ry SHALL be normalized by MAX_RADIUS (377.36)
- **AND** all normalized values SHALL be float32

### Requirement: Preprocessed Dataset Schema
The preprocessed HuggingFace dataset SHALL contain the following columns matching OpenEDS structure with ellipse parameters.

#### Scenario: Dataset columns
- **WHEN** dataset is loaded from HuggingFace (Conner/sddec25-01)
- **THEN** dataset SHALL contain `image` column as uint8[400, 640]
- **AND** dataset SHALL contain `label` column as uint8[400, 640] (binary: 0 or 1)
- **AND** dataset SHALL contain `spatial_weights` column as float32[400, 640]
- **AND** dataset SHALL contain `dist_map` column as float32[2, 400, 640]
- **AND** dataset SHALL contain `cx` column as float32 (normalized 0-1)
- **AND** dataset SHALL contain `cy` column as float32 (normalized 0-1)
- **AND** dataset SHALL contain `rx` column as float32 (normalized)
- **AND** dataset SHALL contain `ry` column as float32 (normalized)
- **AND** dataset SHALL contain `filename` column as string
- **AND** dataset SHALL contain `preprocessed` column as bool (always True)

#### Scenario: Dataset split sizes
- **WHEN** dataset is loaded
- **THEN** train split SHALL contain samples from training subjects
- **AND** validation split SHALL contain samples from validation subjects
- **AND** total samples SHALL be less than original due to empty mask filtering

### Requirement: Preprocessing Detection in Training
Training scripts SHALL detect the `preprocessed` flag and skip redundant gamma correction, CLAHE, and ellipse extraction operations.

#### Scenario: Preprocessed segmentation data loading
- **WHEN** segmentation training script loads a sample with `preprocessed=True`
- **THEN** script SHALL skip gamma correction LUT application
- **AND** script SHALL skip CLAHE application (both CPU and GPU variants)
- **AND** script SHALL apply only stochastic augmentations (RandomHorizontalFlip, Line_augment, Gaussian_blur)

#### Scenario: Preprocessed ellipse data loading
- **WHEN** ellipse training script loads a sample with `preprocessed=True`
- **THEN** script SHALL use precomputed cx, cy, rx, ry columns directly
- **AND** script SHALL skip cv2.findContours and cv2.fitEllipse extraction
- **AND** script SHALL skip gamma correction and CLAHE

#### Scenario: Raw data loading (backward compatibility)
- **WHEN** training script loads a sample with `preprocessed=False` or missing flag
- **THEN** script SHALL apply gamma correction
- **AND** script SHALL apply CLAHE
- **AND** script SHALL extract ellipse parameters at runtime
- **AND** script SHALL apply stochastic augmentations
