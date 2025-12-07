# Training Specification

## Purpose

Define requirements for training scripts that train neural network models for eye pupil segmentation on the OpenEDS dataset, including model architectures, loss functions, and MLflow integration.

## Requirements

### Requirement: EfficientViT Training Script
The system SHALL provide a training script `train_efficientvit.py` that trains a TinyEfficientViT model for eye pupil segmentation on the preprocessed OpenEDS dataset.

#### Scenario: Training script execution
- **WHEN** user runs `modal run training/train_efficientvit.py`
- **THEN** training completes using preprocessed dataset from HuggingFace
- **AND** logs metrics to MLflow
- **AND** best model is exported to ONNX format

#### Scenario: Model parameter constraint
- **WHEN** TinyEfficientViTSeg model is instantiated
- **THEN** total trainable parameters SHALL be less than 60,000

#### Scenario: Dataset loading
- **WHEN** training script loads dataset
- **THEN** script SHALL load from `Conner/sddec25-01` HuggingFace repository
- **AND** script SHALL use preprocessed images directly (no runtime gamma/CLAHE)
- **AND** script SHALL use precomputed spatial_weights and dist_map

### Requirement: TinyEfficientViT Model Architecture
The system SHALL implement a TinyEfficientViTSeg model based on EfficientViT-MSRA architecture with drastically reduced dimensions to meet the <60k parameter constraint.

#### Scenario: Model configuration
- **WHEN** TinyEfficientViTSeg is created with default config
- **THEN** embed_dim SHALL be approximately (8, 16, 24)
- **AND** depth SHALL be (1, 1, 1)
- **AND** num_heads SHALL be (1, 1, 2)

#### Scenario: Forward pass
- **WHEN** model receives input tensor of shape (B, 1, 400, 640)
- **THEN** output tensor SHALL have shape (B, 2, 400, 640)

### Requirement: Segmentation Decoder
The system SHALL implement a lightweight decoder that produces dense predictions from multi-scale encoder features.

#### Scenario: Decoder with skip connections
- **WHEN** encoder produces features at multiple scales
- **THEN** decoder SHALL upsample and combine features to produce full-resolution output

### Requirement: MLflow Integration
The system SHALL log training metrics, parameters, and artifacts to MLflow.

#### Scenario: Metric logging
- **WHEN** each epoch completes
- **THEN** train_loss, valid_loss, train_iou, valid_iou SHALL be logged
- **AND** loss components (ce_loss, dice_loss, surface_loss) SHALL be logged

#### Scenario: Model tagging
- **WHEN** MLflow run is created
- **THEN** model_type tag SHALL be "TinyEfficientViT"
- **AND** architecture tag SHALL distinguish from ShallowNet runs

### Requirement: Ellipse Model PyTorch Checkpoint Export
The system SHALL export the trained ellipse regression model to PyTorch checkpoint format compatible with edge deployment and fine-tuning.

#### Scenario: PyTorch checkpoint export
- **WHEN** training completes with best validation mIoU
- **THEN** best model SHALL be saved to `best_ellipse_model.pt`
- **AND** checkpoint SHALL contain model `state_dict()`
- **AND** model SHALL be in contiguous memory format for portability

#### Scenario: Epoch checkpoints
- **WHEN** training reaches checkpoint epochs (every 10 epochs or final epoch)
- **THEN** checkpoint SHALL be saved as `ellipse_model_epoch_{n}.pt`
- **AND** all checkpoints SHALL be uploaded to MLflow as artifacts

#### Scenario: Compiled model handling
- **WHEN** model is wrapped with `torch.compile()`
- **THEN** export function SHALL unwrap to `_orig_mod` before saving
- **AND** checkpoint SHALL contain unwrapped model weights

### Requirement: Ellipse Regression Colab Notebook
The system SHALL provide a Jupyter notebook `train_ellipse.ipynb` that trains an EllipseRegressionNet model for pupil ellipse parameter prediction, runnable in Google Colab with free GPU.

#### Scenario: Notebook execution in Colab
- **WHEN** user opens `training/train_ellipse.ipynb` in Google Colab
- **AND** user selects GPU runtime
- **AND** user runs all cells
- **THEN** training completes successfully
- **AND** trained model is exported to ONNX format

#### Scenario: Dataset loading from HuggingFace
- **WHEN** notebook executes dataset loading cells
- **THEN** OpenEDS dataset SHALL be downloaded from `Conner/sddec25-01`
- **AND** train and validation splits SHALL be available
- **AND** precomputed ellipse parameters (cx, cy, rx, ry) SHALL be available

#### Scenario: Model architecture consistency
- **WHEN** EllipseRegressionNet is instantiated in notebook
- **THEN** model architecture SHALL match `train_ellipse.py` Modal version
- **AND** model SHALL output 4 parameters (cx, cy, rx, ry)

#### Scenario: Training visualization
- **WHEN** training completes
- **THEN** notebook SHALL display loss curves
- **AND** notebook SHALL display sample prediction visualizations
- **AND** notebook SHALL report final mIoU and error metrics

### Requirement: Colab Runtime Configuration
The notebook SHALL include setup cells that configure the Colab environment for GPU training.

#### Scenario: GPU availability check
- **WHEN** user runs the setup cells
- **THEN** notebook SHALL verify GPU is available
- **AND** notebook SHALL print GPU name and memory

#### Scenario: Dependency installation
- **WHEN** user runs the pip install cell
- **THEN** required packages SHALL be installed (torch, torchvision, opencv-python, datasets, pillow, scikit-learn, tqdm, matplotlib, onnx)

### Requirement: Model Export from Colab
The notebook SHALL provide functionality to export and download the trained model.

#### Scenario: ONNX export
- **WHEN** training completes with best validation mIoU
- **THEN** notebook SHALL export model to `best_ellipse_model.onnx`
- **AND** notebook SHALL display file size

#### Scenario: Model download
- **WHEN** user runs the download cell
- **THEN** trained ONNX model SHALL be downloadable from Colab

### Requirement: EfficientViT Segmentation Colab Notebook
The system SHALL provide a Jupyter notebook `train_efficientvit.ipynb` that trains a TinyEfficientViTSeg model for eye pupil segmentation, runnable in Google Colab with free GPU.

#### Scenario: Notebook execution in Colab
- **WHEN** user opens `training/train_efficientvit.ipynb` in Google Colab
- **AND** user selects GPU runtime
- **AND** user runs all cells
- **THEN** training completes successfully
- **AND** trained model is saved as PyTorch checkpoint

#### Scenario: Dataset loading from HuggingFace
- **WHEN** notebook executes dataset loading cells
- **THEN** OpenEDS dataset SHALL be downloaded from `Conner/openeds-precomputed`
- **AND** train and validation splits SHALL be available

#### Scenario: Model architecture consistency
- **WHEN** TinyEfficientViTSeg is instantiated in notebook
- **THEN** model architecture SHALL match `train_efficientvit.py` Modal version
- **AND** model SHALL have less than 60,000 trainable parameters
- **AND** model SHALL output shape (B, 2, 400, 640) for input shape (B, 1, 400, 640)

#### Scenario: Training visualization
- **WHEN** training completes
- **THEN** notebook SHALL display loss curves (total, CE, dice, surface)
- **AND** notebook SHALL display sample prediction visualizations
- **AND** notebook SHALL report final mIoU and per-class IoU

### Requirement: EfficientViT Model Export
The EfficientViT notebook SHALL provide functionality to export and download the trained segmentation model.

#### Scenario: PyTorch checkpoint save
- **WHEN** training completes with best validation mIoU
- **THEN** notebook SHALL save model to `best_efficientvit_model.pt`
- **AND** notebook SHALL display file size

#### Scenario: ONNX export
- **WHEN** user runs the ONNX export cell
- **THEN** notebook SHALL export model to `best_efficientvit_model.onnx`
- **AND** ONNX file size SHALL be less than 300KB

#### Scenario: Model download
- **WHEN** user runs the download cell
- **THEN** trained model files SHALL be downloadable from Colab

### Requirement: Ultra-Tiny EfficientViT Training Script
The system SHALL provide a training script `train_efficientvit_tiny_local.py` that trains a TinyEfficientViT-Micro model with less than 10,000 parameters for eye pupil segmentation on the OpenEDS dataset.

#### Scenario: Training script execution
- **WHEN** user runs `python training/train_efficientvit_tiny_local.py`
- **THEN** training completes and logs metrics to MLflow
- **AND** best model is saved to `best_efficientvit_tiny_model.pt`

#### Scenario: Distinct checkpoint filenames
- **WHEN** training saves model checkpoints
- **THEN** best model SHALL be saved as `best_efficientvit_tiny_model.pt`
- **AND** epoch checkpoints SHALL be saved as `efficientvit_tiny_model_epoch_{n}.pt`
- **AND** filenames SHALL NOT conflict with standard EfficientViT checkpoints

#### Scenario: Model parameter constraint
- **WHEN** TinyEfficientViT-Micro model is instantiated
- **THEN** total trainable parameters SHALL be less than 10,000

### Requirement: TinyEfficientViT-Micro Model Architecture
The system SHALL implement a TinyEfficientViT-Micro model with drastically reduced dimensions to meet the <10k parameter constraint.

#### Scenario: Model configuration
- **WHEN** TinyEfficientViT-Micro is created with default config
- **THEN** embed_dim SHALL be (8, 12, 18)
- **AND** decoder_dim SHALL be 8
- **AND** num_heads SHALL be (1, 1, 1)

#### Scenario: Forward pass
- **WHEN** model receives input tensor of shape (B, 1, 400, 640)
- **THEN** output tensor SHALL have shape (B, 2, 400, 640)

### Requirement: GPU Training Optimizations
The system SHALL include all GPU optimizations from the standard EfficientViT training script.

#### Scenario: Memory format optimization
- **WHEN** training runs on CUDA device
- **THEN** model SHALL use channels_last memory format
- **AND** torch.compile with max-autotune mode SHALL be applied

#### Scenario: Mixed precision training
- **WHEN** training runs on CUDA device
- **THEN** torch.amp.autocast SHALL be used for forward passes
- **AND** GradScaler SHALL be used for gradient scaling

#### Scenario: CUDA optimizations
- **WHEN** training runs on CUDA device
- **THEN** cudnn.benchmark SHALL be enabled
- **AND** TF32 SHALL be enabled for matmul and cudnn

#### Scenario: GPU metrics accumulation
- **WHEN** training computes epoch metrics
- **THEN** all metrics SHALL be accumulated on GPU
- **AND** GPU-to-CPU transfer SHALL occur only at end of training
- **AND** pre-allocated GPU tensors SHALL be used for metric storage

#### Scenario: Efficient data loading
- **WHEN** DataLoaders are created for CUDA training
- **THEN** pin_memory SHALL be enabled
- **AND** non_blocking transfers SHALL be used

### Requirement: MLflow Tagging for Micro Variant
The system SHALL tag MLflow runs to distinguish TinyEfficientViT-Micro from other model variants.

#### Scenario: Model tagging
- **WHEN** MLflow run is created for micro model training
- **THEN** model_type tag SHALL be "TinyEfficientViT-Micro"
- **AND** tag SHALL distinguish from standard TinyEfficientViT runs

### Requirement: Local EfficientViT Training Script
The system SHALL provide a local training script `train_efficientvit_local.py` that trains a TinyEfficientViT model for eye pupil segmentation on the OpenEDS dataset without requiring Modal cloud infrastructure.

#### Scenario: Local training script execution
- **WHEN** user runs `python training/train_efficientvit_local.py`
- **THEN** training executes on the local machine
- **AND** uses local filesystem for dataset caching
- **AND** saves model checkpoints to local directory

#### Scenario: CLI argument configuration
- **WHEN** user runs `python training/train_efficientvit_local.py --epochs 10 --batch-size 64`
- **THEN** training uses the specified hyperparameters
- **AND** defaults are used for unspecified arguments

#### Scenario: Device selection
- **WHEN** user runs `python training/train_efficientvit_local.py --device cuda`
- **THEN** training runs on the specified device
- **AND** if device is unavailable, script exits with clear error message

#### Scenario: MLflow credentials via environment
- **WHEN** MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID environment variables are set
- **THEN** training logs metrics to the configured MLflow server
- **WHEN** MLflow environment variables are not set
- **THEN** training continues without MLflow logging and prints warning

#### Scenario: Local dataset caching
- **WHEN** dataset is not cached locally
- **THEN** script downloads from HuggingFace and caches to `--data-dir` (default: `~/.cache/openeds`)
- **WHEN** dataset is already cached
- **THEN** script loads from local cache without re-downloading

#### Scenario: Model checkpoint output
- **WHEN** training completes an epoch with improved validation mIoU
- **THEN** best model is saved to `--output-dir` (default: `./checkpoints`)
- **AND** checkpoint filename includes epoch number and validation mIoU

### Requirement: Local Ellipse Regression Training Script
The system SHALL provide a local training script `train_ellipse_local.py` that trains the EllipseRegressionNet model without requiring Modal cloud infrastructure.

#### Scenario: Run training with default parameters
- **WHEN** user executes `python train_ellipse_local.py`
- **THEN** the script downloads the dataset from HuggingFace (if not cached locally)
- **AND** trains the EllipseRegressionNet model for the default number of epochs
- **AND** saves model checkpoints to the local filesystem
- **AND** logs metrics to MLflow (if credentials configured)

#### Scenario: Run training with custom hyperparameters
- **WHEN** user executes `python train_ellipse_local.py --epochs 20 --batch-size 16 --lr 0.0005`
- **THEN** the script uses the specified hyperparameters for training
- **AND** overrides the default values

#### Scenario: Specify local data directory
- **WHEN** user executes `python train_ellipse_local.py --data-dir /path/to/cache`
- **THEN** the script uses the specified directory for dataset caching
- **AND** skips download if dataset already exists at that location

#### Scenario: Run without MLflow credentials
- **WHEN** user executes the script without `MLFLOW_TRACKING_URI` environment variable
- **THEN** the script runs training without MLflow logging
- **AND** prints a warning about disabled MLflow logging
- **AND** still saves model checkpoints locally

### Requirement: Local Device Detection
The local training script SHALL automatically detect and use available compute devices.

#### Scenario: CUDA GPU available
- **WHEN** the system has a CUDA-capable GPU
- **THEN** training runs on the GPU by default

#### Scenario: CPU-only system
- **WHEN** no CUDA GPU is available
- **THEN** training falls back to CPU
- **AND** prints a warning about expected slower performance

#### Scenario: User forces CPU execution
- **WHEN** user executes `python train_ellipse_local.py --device cpu`
- **THEN** training runs on CPU regardless of GPU availability

### Requirement: Local Dataset Caching
The local training script SHALL cache the dataset to avoid repeated downloads.

#### Scenario: First run downloads dataset
- **WHEN** the dataset is not present in the cache directory
- **THEN** the script downloads from HuggingFace Hub
- **AND** saves to the local cache directory
- **AND** prints download progress

#### Scenario: Subsequent runs use cache
- **WHEN** the dataset exists in the cache directory
- **THEN** the script loads from local cache without network access
- **AND** prints confirmation of cache usage

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

### Requirement: Ellipse Training Application
The system SHALL provide a training application `apps/train_ellipse/` that trains an `EllipseRegressionNet` model for pupil ellipse parameter prediction, using the model from `packages/ellipse/`.

#### Scenario: CLI execution
- **WHEN** user runs `train-ellipse --epochs 15 --batch-size 4`
- **THEN** training executes using specified hyperparameters
- **AND** model is imported from `ellipse` package
- **AND** checkpoints are saved to specified output directory

#### Scenario: Dataset loading
- **WHEN** training loads dataset
- **THEN** it SHALL load from `Conner/sddec25-01` HuggingFace repository
- **AND** it SHALL use precomputed ellipse parameters (cx, cy, rx, ry) from dataset
- **AND** it SHALL apply augmentations (line augment, gaussian blur, horizontal flip)

#### Scenario: Model import from package
- **WHEN** training script initializes model
- **THEN** it SHALL import `EllipseRegressionNet` from `ellipse` package
- **AND** it SHALL import `EllipseRegressionLoss` from `ellipse` package
- **AND** no model definition code SHALL exist in the training app

#### Scenario: MLflow integration
- **WHEN** MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID are set
- **THEN** training SHALL log metrics, parameters, and artifacts to MLflow
- **AND** model_type tag SHALL be "EllipseRegressionNet"

### Requirement: Inference-Aligned Ellipse Training Application
The system SHALL provide a training application `apps/train_ellipse_inf/` that trains on raw images without preprocessing, so the trained model works directly on camera input.

#### Scenario: CLI execution
- **WHEN** user runs `train-ellipse-inf --epochs 15 --batch-size 4`
- **THEN** training executes on raw images (no gamma correction, no CLAHE)
- **AND** model is imported from `ellipse` package

#### Scenario: Raw image training
- **WHEN** training loads images from dataset
- **THEN** images SHALL be used directly without preprocessing
- **AND** only stochastic augmentations (line, blur, flip) SHALL be applied
- **AND** model learns to handle raw camera input

#### Scenario: Dataset source
- **WHEN** inference-aligned training loads dataset
- **THEN** it SHALL load from `Conner/sddec25-01` HuggingFace repository
- **AND** it SHALL use the same precomputed ellipse parameters as standard training

### Requirement: Ellipse Training App pyproject.toml Configuration
The training applications SHALL follow the established monorepo pattern with workspace dependencies.

#### Scenario: Standard app dependencies
- **WHEN** `apps/train_ellipse/pyproject.toml` is inspected
- **THEN** it SHALL include `ellipse` as workspace dependency
- **AND** it SHALL include all required training dependencies (torch, datasets, mlflow, etc.)
- **AND** it SHALL define CLI entry point `train-ellipse = "train_ellipse.main:main"`

#### Scenario: Inference-aligned app dependencies
- **WHEN** `apps/train_ellipse_inf/pyproject.toml` is inspected
- **THEN** it SHALL include `ellipse` as workspace dependency
- **AND** it SHALL define CLI entry point `train-ellipse-inf = "train_ellipse_inf.main:main"`

### Requirement: GPU Training Optimizations for Ellipse Apps
The ellipse training applications SHALL include GPU optimizations consistent with existing training apps.

#### Scenario: Memory format optimization
- **WHEN** training runs on CUDA device
- **THEN** model SHALL use channels_last memory format
- **AND** model SHALL optionally use torch.compile

#### Scenario: Mixed precision training
- **WHEN** training runs on CUDA device
- **THEN** torch.amp.autocast SHALL be used for forward passes
- **AND** GradScaler SHALL be used for gradient scaling

#### Scenario: GPU metrics accumulation
- **WHEN** training computes epoch metrics
- **THEN** metrics SHALL be accumulated on GPU
- **AND** GPU-to-CPU transfer SHALL occur at end of epoch
