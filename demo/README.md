# VisionAssist Live Demo

Real-time semantic segmentation demonstration for the VisionAssist medical assistive technology project. This application performs live eye tracking and facial feature detection with webcam input, showcasing the capabilities developed for autonomous wheelchair safety monitoring.

## Available Demos

This directory contains two demo applications, each supporting different model architectures:

| Script | Model | Description |
|--------|-------|-------------|
| `demo_shallownet.py` | ShallowNet | Demo using ShallowNet CNN model |
| `demo_efficientvit.py` | TinyEfficientViT | Demo using TinyEfficientViT transformer model |

### Which Demo Should I Use?

- **`demo_shallownet.py`** - Use this with ShallowNet `.pt` checkpoint files
- **`demo_efficientvit.py`** - Use this with TinyEfficientViT `.pt` checkpoint files (from `training/train_efficientvit.py`)

Both demos use PyTorch for inference and support the same device backends (CUDA, MPS, CPU) with identical command-line interfaces.

## Overview

The live demos capture video from a 1080p webcam, detect facial landmarks using MediaPipe, extract the left eye region, and perform semantic segmentation inference using PyTorch. Results are visualized with a green overlay indicating pupil detection, along with real-time performance metrics.

**Key Features**:
- Real-time semantic segmentation at 60 FPS (GPU)
- MediaPipe facial landmark detection for eye region extraction
- Preprocessing pipeline matching training exactly (gamma correction, CLAHE, normalization)
- Visual feedback with status indicators and performance metrics
- GPU acceleration with CUDA (NVIDIA), MPS (Apple Silicon), or CPU fallback
- Multiple model support: ShallowNet and TinyEfficientViT architectures

## Prerequisites

### Hardware Requirements

- **Webcam**: 1080p resolution strongly recommended
  - Eye region must be at least 100x100 pixels for reliable detection
  - Lower resolution cameras may work but require closer positioning
- **GPU**: NVIDIA GPU with CUDA support strongly recommended
  - Real-time performance (60 FPS) requires GPU acceleration
  - CPU-only mode works but may achieve 5-15 FPS depending on hardware
- **Memory**: Minimum 4GB RAM (3.2GB typical usage on embedded systems)

### Software Requirements

- **Python**: 3.10 - 3.12 (MediaPipe does not support Python 3.13+)
- **uv**: Python package manager (included in flake.nix devshell)
- **Operating System**: Linux, macOS, or Windows with CUDA support (for GPU)

### Trained Model

A trained PyTorch model file (.pt) is required to run the demo. The model must have:
- **Input shape**: `(1, 1, 400, 640)` - single-channel grayscale, 400x640 resolution (PyTorch tensor)
- **Output shape**: `(1, 2, 400, 640)` - two-channel logits (background, pupil) (PyTorch tensor)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/sddec25-01-dd.git
cd sddec25-01-dd/demo
```

### 2. Enter the Nix Development Shell (Recommended)

If using the project's Nix flake (recommended):

```bash
# From repository root
nix develop

# Or if using direnv
cd sddec25-01-dd  # direnv will auto-activate
```

### 3. Install Python Dependencies with uv

```bash
# Install all dependencies (supports CUDA, MPS, and CPU)
uv sync
```

**Dependencies** (managed by `pyproject.toml`):
- `opencv-python>=4.5.0` - Computer vision and image processing
- `torch>=2.0.0` - PyTorch with CUDA and MPS support
- `mediapipe>=0.10.0` - Facial landmark detection
- `numpy>=1.21.0` - Numerical operations

**Note on Device Support**:
- **NVIDIA GPU (CUDA)**: Automatically detected if available
- **Apple Silicon (MPS)**: Automatically detected on M1/M2/M3 Macs
- **CPU**: Fallback when no GPU is available
- Use `--device cuda|mps|cpu` to force a specific device

### 4. Verify Installation

```bash
uv run python demo_shallownet.py --help
# or
uv run python demo_efficientvit.py --help
```

Expected output (both demos have identical interfaces):
```
usage: demo_shallownet.py [-h] --model MODEL [--camera CAMERA] [--device DEVICE] [--verbose]

VisionAssist Live Demo - ShallowNet Semantic Segmentation

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    Path to PyTorch model file (REQUIRED)
  --camera CAMERA  Camera index (default: 0)
  --device DEVICE  Force device (cuda, mps, cpu) (default: auto)
  --verbose        Enable comprehensive logging
```

## Usage

### ShallowNet Demo (demo_shallownet.py)

Use `demo_shallownet.py` for ShallowNet model checkpoints:

```bash
# Basic usage
uv run python demo_shallownet.py --model path/to/shallownet_model.pt

# With camera selection
uv run python demo_shallownet.py --model path/to/shallownet_model.pt --camera 1

# With verbose logging
uv run python demo_shallownet.py --model path/to/shallownet_model.pt --verbose

# Force MPS on Apple Silicon
uv run python demo_shallownet.py --model path/to/shallownet_model.pt --device mps
```

### TinyEfficientViT Demo (demo_efficientvit.py)

Use `demo_efficientvit.py` for TinyEfficientViT model checkpoints (from `training/train_efficientvit.py`):

```bash
# Basic usage
uv run python demo_efficientvit.py --model path/to/efficientvit_model.pt

# With camera selection
uv run python demo_efficientvit.py --model path/to/efficientvit_model.pt --camera 1

# With verbose logging
uv run python demo_efficientvit.py --model path/to/efficientvit_model.pt --verbose

# Force MPS on Apple Silicon
uv run python demo_efficientvit.py --model path/to/efficientvit_model.pt --device mps
```

**Note**: The TinyEfficientViT model has different architecture parameters than ShallowNet:
- Input shape: `(1, 1, 400, 640)` - same as ShallowNet
- Output shape: `(1, 2, 400, 640)` - same as ShallowNet
- Architecture: Transformer-based with ~50k parameters (vs CNN-based ShallowNet)

### Common Options

Camera indices typically start at 0 (default webcam). Use `--camera 1` for external webcams.

Verbose mode prints:
- Frame capture times
- Face detection results
- Preprocessing step timings (grayscale, gamma, CLAHE, resize, normalize)
- Inference times
- Mask statistics

### Full Examples

```bash
# ShallowNet with external webcam and verbose logging
uv run python demo_shallownet.py --model ../training/models/shallownet_epoch50.pt --camera 1 --verbose

# TinyEfficientViT with external webcam and verbose logging
uv run python demo_efficientvit.py --model ../training/best_efficientvit_model.pt --camera 1 --verbose
```

## Command-Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model` | string | **Yes** | - | Path to trained PyTorch model file (.pt) |
| `--camera` | integer | No | `0` | Camera device index (0 for default webcam) |
| `--device` | string | No | `auto` | Force device ("cuda", "mps", "cpu") |
| `--verbose` | flag | No | `False` | Enable comprehensive logging for debugging |

## Keyboard Controls

| Key | Action |
|-----|--------|
| **ESC** | Exit the demo application |
| **SPACE** | Pause/resume processing (frame display freezes) |

## Display Elements

The demo window shows the following visual elements:

### Status Banner (Top Center)
- **"Face Detected"** (Green) - Face successfully detected, segmentation active
- **"No Face Detected"** (Yellow) - Move into camera view
- **"Move Closer"** (Yellow) - Face detected but eye region too small (<100px)

### Segmentation Overlay
- **Green overlay** - Indicates detected pupil region (semantic segmentation output)
- **Transparency**: 50% blend with original frame for visibility

### Eye Bounding Box
- **Green rectangle** - Marks the extracted left eye region used for inference
- Includes 20% padding around MediaPipe eye landmarks
- Maintains 1.6:1 aspect ratio (640:400) matching model input

### Performance Metrics (Top-Left)
- **FPS** - Frames per second (rolling 30-frame average)
- **Inference** - Model inference time in milliseconds
- **Device** - Execution provider (GPU or CPU)

## Optimal Conditions

For best results, ensure the following environmental conditions:

### Lighting
- **Good illumination**: Bright, even lighting on the face
- **Avoid glare**: Position lighting to minimize reflections on glasses or eyes
- **Natural or neutral light**: Avoid extreme color temperature (very warm or cool lighting)

### Distance
- **Eye region size**: Aim for >100 pixels width/height in the bounding box
- **Typical distance**: 1-2 feet (30-60 cm) from a 1080p webcam
- If "Move Closer" appears frequently, reduce distance to camera

### Angle
- **Face camera**: Look directly at the webcam for best facial landmark detection
- **Avoid extreme angles**: Side profiles or tilted heads reduce detection accuracy
- **Eye visibility**: Ensure the left eye is clearly visible (not obscured by hair, glasses glare, etc.)

### Camera Position
- **Eye level**: Position webcam at or slightly above eye level
- **Stable mount**: Use a tripod or stable surface to minimize motion blur
- **1080p resolution**: Configure webcam for 1920x1080 resolution (script attempts this automatically)

## Troubleshooting

### No Camera Detected

**Error**: `Failed to open camera 0. Check that the camera is connected and not in use.`

**Solutions**:
1. Verify the camera is physically connected (USB webcam) or enabled (built-in)
2. Check if another application is using the camera (Zoom, Teams, Skype, etc.)
3. Try a different camera index: `--camera 1` or `--camera 2`
4. On Linux, verify permissions: `ls -l /dev/video*` and ensure your user has access
5. Test camera with another tool: `cheese` (Linux), `Photo Booth` (macOS), `Camera` app (Windows)

### Low FPS on CPU

**Symptom**: FPS < 15, inference time > 50ms

**Solutions**:
1. **Check GPU availability**:
   ```bash
   # Check CUDA (NVIDIA)
   uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"

   # Check MPS (Apple Silicon)
   uv run python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
   ```

2. **Force GPU if available**:
   ```bash
   uv run python demo_shallownet.py --model model.pt --device cuda  # NVIDIA
   uv run python demo_shallownet.py --model model.pt --device mps   # Apple Silicon
   ```

3. **Reduce resolution**: Use a lower resolution webcam (720p) to reduce preprocessing overhead
4. **Close background applications**: Free up CPU resources by closing unnecessary programs
5. **Upgrade hardware**: Consider a faster CPU or adding a dedicated GPU

**Expected performance**:
- **GPU (NVIDIA RTX 3060 or similar)**: 60+ FPS, 5-10ms inference
- **MPS (Apple M1/M2/M3)**: 30-60 FPS, 10-20ms inference
- **CPU (Intel i7 or AMD Ryzen 7)**: 10-20 FPS, 50-100ms inference
- **CPU (Older hardware)**: 5-10 FPS, 100-200ms inference

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'cv2'` (or `torch`, `mediapipe`)

**Solutions**:
1. Ensure you synced dependencies: `uv sync`
2. Run commands with `uv run` prefix: `uv run python demo_shallownet.py ...`
3. Check Python version: `uv run python --version` (must be 3.10-3.12)
4. Re-sync dependencies: `rm -rf .venv && uv sync`

### GPU/MPS Setup Issues

**Symptom**: Device shows CPU when you expected GPU or MPS

**Check Available Devices**:
```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

**Solutions for NVIDIA GPU (CUDA)**:
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA support: `uv run python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure compatible PyTorch version with your CUDA version

**Solutions for Apple Silicon (MPS)**:
1. Requires macOS 12.3+ and Python 3.10-3.12
2. Check MPS availability: `uv run python -c "import torch; print(torch.backends.mps.is_available())"`
3. If MPS is available but not working, try: `--device mps`

**Force a Specific Device**:
```bash
uv run python demo_shallownet.py --model model.pt --device mps  # Force MPS
uv run python demo_shallownet.py --model model.pt --device cuda  # Force CUDA
uv run python demo_shallownet.py --model model.pt --device cpu   # Force CPU
```

### Model Loading Errors

**Error**: `RuntimeError: Error loading model.pt`

**Cause**: The model file is not a valid PyTorch state dict.

**Solutions**:
1. Verify you are using a ShallowNet model trained for VisionAssist (640x400 input)
2. Check model format - must be a PyTorch state dict (.pt), not ONNX
3. Verify file integrity: ensure the .pt file was not corrupted during download
4. Test model loading:
   ```bash
   uv run python -c "
   import torch
   model = torch.load('model.pt', map_location='cpu')
   print('Model loaded successfully')
   print('Keys:', list(model.keys())[:5])
   "
   ```

5. Retrain the model with correct input dimensions or use a compatible model

### Face Detection Issues

**Symptom**: "No Face Detected" appears constantly despite being in view

**Solutions**:
1. **Improve lighting**: Ensure face is well-lit and evenly illuminated
2. **Face camera**: Look directly at the webcam, avoid extreme side angles
3. **Move closer**: MediaPipe works best at 1-2 feet (30-60 cm) distance
4. **Remove obstructions**: Ensure face is not obscured by hair, hands, or objects
5. **Check camera resolution**: Verify camera is operating at 1080p (script prints actual resolution at startup with `--verbose`)

## Technical Notes

### Preprocessing Pipeline

The preprocessing pipeline is designed to match the training pipeline exactly. Any deviation will result in degraded model performance. The steps are:

1. **Grayscale conversion**: BGR to grayscale using OpenCV's `cvtColor`
2. **Gamma correction**: Apply gamma=0.8 via lookup table for brightness adjustment
3. **CLAHE**: Contrast Limited Adaptive Histogram Equalization (clip=1.5, tile=8x8)
4. **Resize**: Bilinear interpolation to 640x400 (model input resolution)
5. **Normalization**: `(pixel/255.0 - 0.5) / 0.5` to range [-1, 1]
6. **Tensor format**: Add batch and channel dimensions to `(1, 1, 400, 640)`

**Critical**: Do not modify preprocessing without retraining the model.

### Left Eye Only

The demo processes the **left eye only** by design. This is consistent with the training dataset and model architecture. The left eye is extracted using MediaPipe's 12 facial landmarks specific to the left eye region (indices: 362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466).

### Model Input Resolution

The model expects **640x400** grayscale input. This aspect ratio (1.6:1) is enforced during eye region extraction by:
1. Computing a bounding box around MediaPipe left eye landmarks
2. Adding 20% padding on each side
3. Expanding the bounding box to 1.6:1 aspect ratio (expanding width or height as needed)
4. Resizing the extracted region to exactly 640x400 pixels

### Segmentation Output

The model outputs two-channel logits as a PyTorch tensor `(1, 2, 400, 640)`:
- **Channel 0**: Background probability
- **Channel 1**: Pupil probability

The demo applies `argmax` to produce a binary mask where:
- **0** = Background (no overlay)
- **1** = Pupil (green overlay)

PyTorch tensors are automatically moved to the appropriate device (CPU/CUDA/MPS) and converted to NumPy arrays for visualization.

### Performance Metrics

- **FPS (Frames Per Second)**: Rolling average over 30 frames
- **Inference Time**: Time spent in PyTorch model forward pass, excluding preprocessing and post-processing
- **Total Pipeline**: Includes frame capture, face detection, preprocessing, inference, visualization
- **Device**: Shows current execution device (cuda, mps, or cpu)

Typical breakdown on GPU:
- Frame capture: ~1-2ms
- Face detection (MediaPipe): ~5-10ms
- Preprocessing: ~2-3ms
- Inference: ~5-10ms
- Visualization: ~1-2ms
- **Total**: ~15-30ms per frame (30-60 FPS)

## Example Usage Scenarios

### Academic Presentation Demo

```bash
# High-quality ShallowNet demo for presentation with verbose logging
uv run python demo_shallownet.py --model ../models/shallownet_final.pt --verbose

# High-quality EfficientViT demo for presentation with verbose logging
uv run python demo_efficientvit.py --model ../models/efficientvit_final.pt --verbose
```

Expected output with `--verbose`:
```
================================================================================
VisionAssist Live Demo
================================================================================
Model: ../models/shallownet_final.pt
Camera: 0
Device: cuda

Controls:
  ESC - Exit
  SPACE - Pause/Resume
================================================================================

Initializing camera 0...
Camera initialized: 1920x1080 (requested 1920x1080)
Loading PyTorch model from ../models/shallownet_final.pt...
Using device: cuda
Model validated: input (1, 1, 400, 640), output (1, 2, 400, 640)
Initializing MediaPipe Face Mesh...
MediaPipe Face Mesh initialized
Preprocessing initialized: gamma=0.8, CLAHE(clip=1.5, tile=(8, 8))

Frame 0: capture 1.23ms
  Face detection: 8.45ms, detected=True
  First 3 eye landmarks: [(1234, 567), (1245, 560), (1250, 555)]
  Grayscale: 0.15ms, shape=(180, 288), range=[10, 245]
  Gamma correction: 0.12ms, range=[8, 238]
  CLAHE: 1.85ms, range=[0, 255]
  Resize: 0.68ms, shape=(400, 640)
  Normalize: 0.22ms, range=[-0.961, 0.922]
  Final tensor shape: (1, 1, 400, 640), total preprocessing: 3.02ms
  Inference: 6.8ms, output shape=(1, 2, 400, 640), mask shape=(400, 640), mask values=[0, 1]
```

### Quick Performance Test

```bash
# Run for 30 seconds and check FPS
uv run python demo_shallownet.py --model model.pt
# Press ESC after 30 seconds, observe FPS counter
```

### Debugging Low Accuracy

```bash
# Enable verbose mode to inspect preprocessing values
uv run python demo_shallownet.py --model model.pt --verbose > debug.log 2>&1
# Review debug.log for anomalies in preprocessing ranges
```

### Multi-Camera Testing

```bash
# Test each camera index to find the correct 1080p webcam
uv run python demo_shallownet.py --model model.pt --camera 0  # Built-in
uv run python demo_shallownet.py --model model.pt --camera 1  # External USB
uv run python demo_shallownet.py --model model.pt --camera 2  # Secondary external
```

## Project Context

This live demo is part of the **VisionAssist** senior design project (SDDEC25-01) at Iowa State University. The project focuses on optimizing AI-powered eye tracking systems for real-time medical monitoring and assistive technology, specifically for individuals with mobility impairments using powered wheelchairs.

**Application Domain**: Medical assistive technology for early detection of medical episodes (epilepsy, cardiovascular distress) through eye movement and posture analysis, enabling autonomous safety responses.

For more information, see the project documentation at `/home/connerohnesorge/Documents/001Repos/sddec25-01-dd/`.

## License

Iowa State University Senior Design Project (SDDEC25-01) - Academic Use Only

## Contact

For questions or issues, contact the VisionAssist development team via the project repository or course instructor.
