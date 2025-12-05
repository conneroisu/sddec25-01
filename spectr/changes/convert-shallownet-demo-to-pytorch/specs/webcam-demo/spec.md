# Webcam Demo Specification Delta

## ADDED Requirements

### Requirement: PyTorch Model Inference
The system SHALL load the trained ShallowNet PyTorch model and perform real-time inference with multi-device support including Apple Silicon MPS acceleration.

#### Scenario: Model loaded from PyTorch state dict
- **WHEN** the application starts with a valid `.pt` model path via `--model` flag
- **THEN** the ShallowNet model SHALL be instantiated with matching architecture
- **AND** load state dict from the `.pt` file
- **AND** set model to eval mode on the target device

#### Scenario: Auto device detection (default)
- **WHEN** `--device auto` or no device flag is specified
- **THEN** the system SHALL check for CUDA availability first
- **AND** check for MPS availability second (Apple Silicon)
- **AND** fall back to CPU if neither GPU backend is available
- **AND** log which device is selected on startup

#### Scenario: MPS acceleration on Apple Silicon
- **WHEN** running on macOS with Apple Silicon (M1/M2/M3)
- **AND** `--device mps` or auto-detection selects MPS
- **THEN** the model SHALL run on the MPS backend
- **AND** inference SHALL be GPU-accelerated via Metal Performance Shaders

#### Scenario: Explicit device override
- **WHEN** user specifies `--device cuda`, `--device mps`, or `--device cpu`
- **THEN** the system SHALL attempt to use the specified device
- **AND** fall back to CPU with a warning if the requested device is unavailable

#### Scenario: Inference performed with PyTorch
- **WHEN** a preprocessed eye tensor is ready
- **THEN** the system SHALL convert NumPy array to PyTorch tensor on device
- **AND** run inference within `torch.no_grad()` context
- **AND** apply argmax to get binary segmentation mask
- **AND** convert output back to NumPy for visualization
- **AND** display inference time in milliseconds

#### Scenario: Model file not found or invalid
- **WHEN** the specified `.pt` model file does not exist or cannot be loaded
- **THEN** the application SHALL display a clear error message
- **AND** exit gracefully with instructions to provide a valid ShallowNet `.pt` model
