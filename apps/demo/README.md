# VisionAssist Live Demo

Real-time semantic segmentation demonstration for the VisionAssist medical assistive technology project. This application performs live eye tracking and facial feature detection using the ShallowNet model with webcam input, showcasing the capabilities developed for autonomous wheelchair safety monitoring.

## Overview

The live demo captures video from a 1080p webcam, detects facial landmarks using MediaPipe, extracts the left eye region, and performs semantic segmentation inference using a trained ONNX model. Results are visualized with a green overlay indicating pupil detection, along with real-time performance metrics.

**Key Features**:
- Real-time semantic segmentation at 60 FPS (GPU)
- MediaPipe facial landmark detection for eye region extraction
- Preprocessing pipeline matching training exactly (gamma correction, CLAHE, normalization)
- Visual feedback with status indicators and performance metrics
- GPU acceleration with CUDA (recommended) or CPU fallback

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
- **CUDA Toolkit**: Version 11.x or 12.x (for GPU acceleration)

### Trained Model

A trained ONNX model file is required to run the demo. The model must have:
- **Input shape**: `(1, 1, 400, 640)` - single-channel grayscale, 400x640 resolution
- **Output shape**: `(1, 2, 400, 640)` - two-channel logits (background, pupil)

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
# CPU-only (default)
uv sync

# With GPU support (NVIDIA CUDA)
uv sync --extra gpu
```

**Dependencies** (managed by `pyproject.toml`):
- `opencv-python>=4.5.0` - Computer vision and image processing
- `onnxruntime>=1.17.0` - ONNX Runtime (CPU)
- `onnxruntime-gpu>=1.17.0` - ONNX Runtime with CUDA support (optional, via `--extra gpu`)
- `mediapipe>=0.10.0` - Facial landmark detection
- `numpy>=1.21.0` - Numerical operations

**Note on ONNX Runtime**:
- Use `uv sync --extra gpu` if you have an NVIDIA GPU with CUDA for optimal performance
- The demo automatically selects the best available execution provider (GPU → CPU fallback)
- The lockfile (`uv.lock`) ensures reproducible installs across machines

### 4. Verify Installation

```bash
uv run python demo.py --help
```

Expected output:
```
usage: demo.py [-h] --model MODEL [--camera CAMERA] [--verbose]

VisionAssist Live Demo - ShallowNet Semantic Segmentation

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    Path to ONNX model file (REQUIRED)
  --camera CAMERA  Camera index (default: 0)
  --verbose        Enable comprehensive logging
```

## Usage

### Basic Usage

```bash
uv run python demo.py --model path/to/model.onnx
```

### With Camera Selection

If you have multiple cameras, specify the camera index:

```bash
uv run python demo.py --model path/to/model.onnx --camera 1
```

Camera indices typically start at 0 (default webcam). Use `--camera 1` for external webcams.

### With Verbose Logging

Enable detailed performance logging for debugging:

```bash
uv run python demo.py --model path/to/model.onnx --verbose
```

Verbose mode prints:
- Frame capture times
- Face detection results
- Preprocessing step timings (grayscale, gamma, CLAHE, resize, normalize)
- Inference times
- Mask statistics

### Full Example

```bash
# Run demo with external webcam and verbose logging
uv run python demo.py --model ../training/models/shallownet_epoch50.onnx --camera 1 --verbose
```

## Command-Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model` | string | **Yes** | - | Path to trained ONNX model file |
| `--camera` | integer | No | `0` | Camera device index (0 for default webcam) |
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
1. **Install GPU support**:
   ```bash
   uv sync --extra gpu
   # Verify CUDA is available
   uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   ```
   Expected output should include `'CUDAExecutionProvider'`

2. **Reduce resolution**: Use a lower resolution webcam (720p) to reduce preprocessing overhead
3. **Close background applications**: Free up CPU resources by closing unnecessary programs
4. **Upgrade hardware**: Consider a faster CPU or adding a dedicated GPU

**Expected performance**:
- **GPU (NVIDIA RTX 3060 or similar)**: 60+ FPS, 5-10ms inference
- **CPU (Intel i7 or AMD Ryzen 7)**: 10-20 FPS, 50-100ms inference
- **CPU (Older hardware)**: 5-10 FPS, 100-200ms inference

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'cv2'` (or `onnxruntime`, `mediapipe`)

**Solutions**:
1. Ensure you synced dependencies: `uv sync`
2. Run commands with `uv run` prefix: `uv run python demo.py ...`
3. Check Python version: `uv run python --version` (must be 3.10-3.12)
4. Re-sync dependencies: `rm -rf .venv && uv sync`

### GPU Setup Issues

**Error**: CUDA is installed but demo uses CPU

**Solutions**:
1. **Verify CUDA installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```
   Both commands should succeed and show CUDA/GPU information

2. **Check ONNX Runtime GPU support**:
   ```bash
   uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   ```
   Output should include `'CUDAExecutionProvider'`

3. **Reinstall with GPU support**:
   ```bash
   rm -rf .venv
   uv sync --extra gpu
   ```

4. **CUDA version mismatch**:
   - `onnxruntime-gpu` 1.17.0+ requires CUDA 11.x or 12.x
   - Check compatibility: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

5. **Force CPU for testing** (re-sync without GPU extra):
   ```bash
   rm -rf .venv
   uv sync
   ```

### Model Validation Errors

**Error**: `RuntimeError: Model input shape mismatch. Expected [-1, 1, 400, 640], got ...`

**Cause**: The ONNX model does not match the expected input/output dimensions.

**Solutions**:
1. Verify you are using a ShallowNet model trained for VisionAssist (640x400 input)
2. Check model metadata:
   ```bash
   uv run python -c "
   import onnxruntime as ort
   session = ort.InferenceSession('model.onnx')
   print('Input:', session.get_inputs()[0].shape)
   print('Output:', session.get_outputs()[0].shape)
   "
   ```
   Expected:
   - Input: `[-1, 1, 400, 640]` or `[1, 1, 400, 640]`
   - Output: `[-1, 2, 400, 640]` or `[1, 2, 400, 640]`

3. Retrain the model with correct input dimensions or use a compatible model

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

The model outputs two-channel logits `(1, 2, 400, 640)`:
- **Channel 0**: Background probability
- **Channel 1**: Pupil probability

The demo applies `argmax` to produce a binary mask where:
- **0** = Background (no overlay)
- **1** = Pupil (green overlay)

### Performance Metrics

- **FPS (Frames Per Second)**: Rolling average over 30 frames
- **Inference Time**: Time spent in `session.run()`, excluding preprocessing and post-processing
- **Total Pipeline**: Includes frame capture, face detection, preprocessing, inference, visualization

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
# High-quality demo for presentation with verbose logging
uv run python demo.py --model ../models/shallownet_final.onnx --verbose
```

Expected output with `--verbose`:
```
================================================================================
VisionAssist Live Demo
================================================================================
Model: ../models/shallownet_final.onnx
Camera: 0
Execution Provider: CUDAExecutionProvider

Controls:
  ESC - Exit
  SPACE - Pause/Resume
================================================================================

Initializing camera 0...
Camera initialized: 1920x1080 (requested 1920x1080)
Loading ONNX model from ../models/shallownet_final.onnx...
Using execution provider: CUDAExecutionProvider
Model validated: input [-1, 1, 400, 640], output [-1, 2, 400, 640]
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
uv run python demo.py --model model.onnx
# Press ESC after 30 seconds, observe FPS counter
```

### Debugging Low Accuracy

```bash
# Enable verbose mode to inspect preprocessing values
uv run python demo.py --model model.onnx --verbose > debug.log 2>&1
# Review debug.log for anomalies in preprocessing ranges
```

### Multi-Camera Testing

```bash
# Test each camera index to find the correct 1080p webcam
uv run python demo.py --model model.onnx --camera 0  # Built-in
uv run python demo.py --model model.onnx --camera 1  # External USB
uv run python demo.py --model model.onnx --camera 2  # Secondary external
```

## Project Context

This live demo is part of the **VisionAssist** senior design project (SDDEC25-01) at Iowa State University. The project focuses on optimizing AI-powered eye tracking systems for real-time medical monitoring and assistive technology, specifically for individuals with mobility impairments using powered wheelchairs.

**Application Domain**: Medical assistive technology for early detection of medical episodes (epilepsy, cardiovascular distress) through eye movement and posture analysis, enabling autonomous safety responses.

For more information, see the project documentation at `/home/connerohnesorge/Documents/001Repos/sddec25-01-dd/`.

## License

Iowa State University Senior Design Project (SDDEC25-01) - Academic Use Only

## Contact

For questions or issues, contact the VisionAssist development team via the project repository or course instructor.
