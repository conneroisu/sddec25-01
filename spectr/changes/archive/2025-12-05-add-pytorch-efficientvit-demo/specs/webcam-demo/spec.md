## ADDED Requirements

### Requirement: PyTorch Multi-Device Inference
The system SHALL support native PyTorch inference with automatic device selection across CUDA, MPS, and CPU backends.

#### Scenario: Automatic device detection selects best available
- **WHEN** the application is launched without explicit `--device` argument
- **THEN** the system SHALL check for CUDA availability first
- **AND** if CUDA unavailable, check for MPS availability (Apple Silicon)
- **AND** if MPS unavailable, fall back to CPU
- **AND** display the selected device on startup and in the metrics overlay

#### Scenario: Explicit device selection with CUDA
- **WHEN** the application is launched with `--device cuda`
- **AND** a CUDA-capable GPU is available
- **THEN** the system SHALL use CUDA for inference
- **AND** display "Device: CUDA" in the metrics overlay

#### Scenario: Explicit device selection with MPS
- **WHEN** the application is launched with `--device mps`
- **AND** running on Apple Silicon Mac with MPS available
- **THEN** the system SHALL use MPS for inference
- **AND** display "Device: MPS" in the metrics overlay

#### Scenario: Explicit device selection with CPU
- **WHEN** the application is launched with `--device cpu`
- **THEN** the system SHALL use CPU for inference regardless of GPU availability
- **AND** display "Device: CPU" in the metrics overlay

#### Scenario: Requested device unavailable
- **WHEN** the application is launched with explicit `--device` argument
- **AND** the requested device is not available
- **THEN** the system SHALL display a warning message
- **AND** fall back to the next available device (CUDA → MPS → CPU)
- **AND** continue operation with the fallback device

### Requirement: PyTorch Model Loading
The system SHALL load TinyEfficientViT model weights from PyTorch checkpoint files with validation.

#### Scenario: Valid checkpoint loaded successfully
- **WHEN** the application is launched with `--model path/to/model.pt`
- **AND** the file contains a valid TinyEfficientViT state dict
- **THEN** the system SHALL instantiate the TinyEfficientViT model with correct hyperparameters
- **AND** load the state dict onto the selected device
- **AND** set the model to evaluation mode
- **AND** display "Model loaded: {filename}" on startup

#### Scenario: Checkpoint file not found
- **WHEN** the application is launched with `--model path/to/nonexistent.pt`
- **THEN** the system SHALL display an error message with the expected path
- **AND** exit gracefully with instructions to provide a valid checkpoint

#### Scenario: Incompatible checkpoint (shape mismatch)
- **WHEN** the application is launched with a checkpoint from a different model architecture
- **THEN** the system SHALL catch the state dict loading error
- **AND** display a clear error message explaining the architecture mismatch
- **AND** exit gracefully

### Requirement: TinyEfficientViT Architecture Configuration
The system SHALL instantiate the TinyEfficientViT model with the exact hyperparameters used during training.

#### Scenario: Model instantiated with training configuration
- **WHEN** loading a model for inference
- **THEN** the TinyEfficientViT SHALL be instantiated with:
  - `in_channels=1` (grayscale input)
  - `num_classes=2` (background, pupil)
  - `embed_dims=(16, 32, 64)` (encoder channel dimensions)
  - `depths=(1, 1, 1)` (transformer blocks per stage)
  - `num_heads=(1, 1, 2)` (attention heads per stage)
  - `key_dims=(4, 4, 4)` (attention key dimensions)
  - `attn_ratios=(2, 2, 2)` (attention expansion ratios)
  - `window_sizes=(7, 7, 7)` (local attention window sizes)
  - `mlp_ratios=(2, 2, 2)` (MLP expansion ratios)
  - `decoder_dim=32` (decoder channel dimension)
- **AND** total parameters SHALL be approximately 57,434 (<60k constraint)

### Requirement: PyTorch Preprocessing Pipeline
The system SHALL preprocess eye crops using the exact same pipeline as training to ensure model accuracy.

#### Scenario: Preprocessing applied in training-identical order
- **WHEN** an eye region is cropped from the webcam frame
- **THEN** the system SHALL resize to 640x400 using bilinear interpolation
- **AND** convert to grayscale
- **AND** apply gamma correction with γ=0.8 via lookup table
- **AND** apply CLAHE with clipLimit=1.5 and tileGridSize=(8,8)
- **AND** normalize with mean=0.5 and std=0.5
- **AND** convert to PyTorch tensor of shape (1, 1, 640, 400) float32
- **AND** move tensor to the inference device

#### Scenario: Tensor placed on correct device
- **WHEN** preprocessing produces an input tensor
- **THEN** the tensor SHALL be on the same device as the model (CUDA, MPS, or CPU)
- **AND** no device mismatch errors SHALL occur during inference

### Requirement: PyTorch Inference Execution
The system SHALL run inference using PyTorch best practices for optimal performance.

#### Scenario: Inference with gradient disabled
- **WHEN** running model inference on a preprocessed tensor
- **THEN** the system SHALL use `torch.no_grad()` context manager
- **AND** the model SHALL be in evaluation mode (`model.eval()`)
- **AND** no gradient computation SHALL occur

#### Scenario: Inference produces segmentation mask
- **WHEN** a preprocessed tensor of shape (1, 1, 640, 400) is passed to the model
- **THEN** the model output SHALL have shape (1, 2, 640, 400)
- **AND** argmax over the class dimension SHALL produce a binary mask
- **AND** the mask SHALL be converted to numpy array for visualization
- **AND** inference time SHALL be measured in milliseconds

### Requirement: Command Line Interface
The system SHALL provide command-line arguments for model path, device selection, and camera configuration.

#### Scenario: Required model argument
- **WHEN** the application is launched
- **THEN** the `--model` argument SHALL be required
- **AND** the application SHALL exit with an error if `--model` is not provided

#### Scenario: Optional device argument
- **WHEN** the application is launched
- **THEN** the `--device` argument SHALL be optional with choices: cuda, mps, cpu
- **AND** if not specified, automatic device detection SHALL be used

#### Scenario: Optional camera argument
- **WHEN** the application is launched
- **THEN** the `--camera` argument SHALL be optional
- **AND** the default SHALL be camera index 0

#### Scenario: Optional verbose argument
- **WHEN** the application is launched with `--verbose` flag
- **THEN** comprehensive logging SHALL be enabled including:
  - Device selection details
  - Model loading and parameter count
  - Preprocessing timing
  - Inference timing per frame
  - Memory usage on GPU devices
