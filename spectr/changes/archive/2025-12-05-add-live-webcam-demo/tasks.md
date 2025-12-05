# Implementation Tasks

## 1. Project Structure Setup
- [x] 1.1 Create `demo/` directory in repository root
- [x] 1.2 Create `demo/demo.py` main application file
- [x] 1.3 Create `demo/requirements.txt` for dependencies
- [x] 1.4 Create `demo/README.md` with usage instructions
- [x] 1.5 Add `demo/.gitignore` for model files and cache

## 2. Webcam and Face Detection
- [x] 2.1 Implement webcam capture with OpenCV VideoCapture at 1920x1080
- [x] 2.2 Integrate MediaPipe Face Mesh with tracking mode smoothing enabled
- [x] 2.3 Add camera selection support via `--camera` argument (default: 0)
- [x] 2.4 Implement graceful handling when no face detected (show banner, continue)
- [x] 2.5 Configure MediaPipe: max_num_faces=1, refine_landmarks=True, tracking confidence=0.5

## 3. Eye Region Extraction and Preprocessing
- [x] 3.1 Extract left eye landmarks using 12-point contour (indices: 362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466)
- [x] 3.2 Compute bounding box from landmarks and check size >100px (show "Move Closer" if smaller)
- [x] 3.3 Add 20% padding on all sides
- [x] 3.4 Implement smart crop: expand bounding box to 1.6:1 aspect ratio before extraction
- [x] 3.5 Add grayscale conversion
- [x] 3.6 Implement gamma correction (gamma=0.8) on original crop
- [x] 3.7 Implement CLAHE (clipLimit=1.5, tileGridSize=8x8) on original crop before resize
- [x] 3.8 Add resize to 640x400 with INTER_LINEAR interpolation
- [x] 3.9 Implement normalization (mean=0.5, std=0.5, matching training exactly)

## 4. Model Inference Integration
- [x] 4.1 Implement ONNX Runtime model loading with strict validation
- [x] 4.2 Validate model input shape (1,1,400,640) and output shape (1,2,400,640) on startup
- [x] 4.3 Add required `--model` command-line argument (no default path)
- [x] 4.4 Add GPU execution provider support (CUDAExecutionProvider preferred)
- [x] 4.5 Fallback to CPUExecutionProvider with warning message if GPU unavailable
- [x] 4.6 Log which execution provider is active on startup
- [x] 4.7 Implement inference with proper input tensor format (1,1,400,640) float32
- [x] 4.8 Implement post-processing (argmax for segmentation mask)
- [x] 4.9 Add inference time measurement (milliseconds, 1 decimal place)

## 5. Visualization and Overlay
- [x] 5.1 Implement coordinate transformation (640x400 model space → eye crop → 1080p frame space)
- [x] 5.2 Create green segmentation overlay (pupil=green at 50% alpha, background=transparent)
- [x] 5.3 Keep overlay at model resolution (640x400) and upscale to match eye crop region
- [x] 5.4 Add semi-transparent overlay blending with cv2.addWeighted
- [x] 5.5 Draw bounding box around detected eye region (green, 3-pixel thickness)
- [x] 5.6 Add FPS counter display (top-left, green text, 1.0 font scale)
- [x] 5.7 Add inference time display in ms (top-left, below FPS, 1 decimal)
- [x] 5.8 Add GPU/CPU indicator (top-left, below inference time)
- [x] 5.9 Add top center status banner with semi-transparent black background
- [x] 5.10 Implement status messages: "Face Detected" (green), "No Face Detected" (yellow), "Move Closer" (yellow)

## 6. User Controls and Configuration
- [x] 6.1 Implement keyboard controls (ESC to exit, Space to pause/resume)
- [x] 6.2 Add command-line arguments: --model (required), --camera (default 0), --verbose (optional)
- [x] 6.3 Implement --verbose flag: log everything (timing, shapes, values, coordinates, timestamps)
- [x] 6.4 Create full 1920x1080 display window named "VisionAssist Live Demo"
- [x] 6.5 Implement graceful shutdown and resource cleanup (webcam, ONNX session, windows)

## 7. Documentation
- [x] 7.1 Write `demo/README.md` with setup instructions for 1080p webcam
- [x] 7.2 Document command-line arguments: --model (required), --camera, --verbose
- [x] 7.3 Document keyboard controls: ESC (exit), Space (pause/resume)
- [x] 7.4 Add troubleshooting section (no camera, low FPS on CPU, import errors, GPU setup)
- [x] 7.5 Document hardware requirements: 1080p webcam, GPU strongly recommended
- [x] 7.6 Document optimal conditions: lighting, distance (for >100px eye region), angle
- [x] 7.7 Add example usage commands and expected output

## 8. Testing and Validation
- [x] 8.1 Test with trained ONNX model on 1080p laptop webcam
- [x] 8.2 Verify performance on GPU (target 30+ FPS) and CPU (accept lower FPS)
- [x] 8.3 Test with different lighting conditions (verify CLAHE and gamma help)
- [x] 8.4 Test with multiple faces in frame (verify primary face selection)
- [x] 8.5 Test at various distances (verify "Move Closer" warning when eye <100px)
- [x] 8.6 Test with no face detected (verify banner shows, no crash, clears overlay)
- [x] 8.7 Verify preprocessing EXACTLY matches training pipeline from train.py
- [x] 8.8 Test model validation (try invalid model, verify clear error message)
- [x] 8.9 Test verbose logging (verify all timing, shapes, values logged correctly)
- [x] 8.10 Verify green overlay color, 3px bounding box, top center banner positioning
- [x] 8.11 Test pause/resume functionality (Space key)
- [x] 8.12 Verify segmentation quality and coordinate mapping accuracy
