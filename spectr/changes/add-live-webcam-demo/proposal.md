# Change: Add Live Webcam Demo Application for ShallowNet

## Why

The VisionAssist project needs a real-time demonstration application to showcase the ShallowNet semantic segmentation model trained in `training/`. This demo will be used for academic presentations, poster sessions, and stakeholder demonstrations. Currently, there is no way to visualize the model's eye segmentation capabilities on live webcam input.

The demo must intelligently detect faces, locate eye regions, crop appropriately, and run inference in real-time (30+ FPS) to provide a compelling, interactive demonstration of the AI-powered eye tracking system.

## What Changes

- **ADDED** Live webcam demo application (`demo/` directory)
- **ADDED** Real-time face and eye detection using MediaPipe Face Mesh
- **ADDED** Intelligent eye region cropping and preprocessing pipeline
- **ADDED** ONNX runtime inference integration for ShallowNet model
- **ADDED** Real-time visualization overlay showing pupil segmentation
- **ADDED** Performance monitoring (FPS counter, inference time)
- **ADDED** Model loading and configuration system
- **ADDED** Webcam input handling with multiple camera support
- **ADDED** README documentation for demo setup and usage
- **ADDED** Requirements file for demo dependencies

## Impact

- Affected specs: `webcam-demo` (new capability)
- New directory: `demo/` containing Python application
- Dependencies: OpenCV (4.5+), ONNX Runtime (1.17+), MediaPipe (0.10+), NumPy (1.21+)
- Model artifact: Requires trained ONNX model provided via `--model` flag (user has this)
- Hardware requirements:
  - 1080p webcam (tested with laptop webcam)
  - GPU with CUDA strongly recommended for 1080p real-time processing
  - CPU-only mode supported but may run at lower FPS

## Technical Approach

**Pipeline Architecture:**
1. **Webcam Capture** → OpenCV VideoCapture at 1920x1080 (Full HD)
2. **Face Detection** → MediaPipe Face Mesh with tracking smoothing
3. **Eye Region Extraction** → Extract left eye using 12-point contour, smart crop to 1.6:1 aspect ratio
4. **Preprocessing** → Grayscale → Gamma (0.8) → CLAHE → Resize to 640x400 → Normalize (matching training exactly)
5. **Inference** → ONNX Runtime (prefer GPU, fallback CPU) with strict model validation
6. **Segmentation Overlay** → Map 640x400 predictions to 1080p frame coordinates
7. **Visualization** → Full 1080p display with green overlay, FPS counter, status banner

**Performance Optimizations:**
- Full 1080p native capture for best quality
- MediaPipe Face Mesh with GPU acceleration and built-in tracking smoothing
- ONNX Runtime with CUDA GPU execution provider (fallback to CPU with warning)
- Process all frames (no skipping) - report honest FPS metrics
- Minimum eye region size check (>100px) with "Move Closer" warning
- Strict model validation on startup (fail fast with clear errors)

**User Experience:**
- Single command launch: `python demo/demo.py --model path/to/model.onnx`
- Full 1080p window for impressive presentation quality
- Green semi-transparent pupil segmentation overlay (50% alpha)
- Keyboard controls (ESC to exit, space to pause/resume)
- Top center status banner ("Face Detected", "No Face Detected", "Move Closer")
- On-screen performance metrics (FPS, inference time, GPU/CPU indicator)
- Optional `--verbose` flag for comprehensive debugging logs
- Graceful error handling with clear user feedback

## Rationale

**MediaPipe vs Alternatives:**
- MediaPipe Face Mesh: 30+ FPS, accurate eye landmarks, GPU accelerated ✓
- dlib: Slower (~10 FPS), but very accurate
- OpenCV Haar Cascades: Fast but less accurate for eye detection

MediaPipe is chosen for real-time academic presentation requirements.

**Desktop Window vs Web Interface:**
- Desktop (OpenCV): Lower latency, simpler deployment, better for presentations ✓
- Web: More complexity, network latency, harder setup

Desktop chosen for academic presentation use case.
