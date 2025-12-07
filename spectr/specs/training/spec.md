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
