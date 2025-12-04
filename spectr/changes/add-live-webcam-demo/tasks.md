# Implementation Tasks

## 1. Project Structure Setup
- [ ] 1.1 Create `demo/` directory in repository root
- [ ] 1.2 Create `demo/demo.py` main application file
- [ ] 1.3 Create `demo/requirements.txt` for dependencies
- [ ] 1.4 Create `demo/README.md` with usage instructions
- [ ] 1.5 Add `demo/.gitignore` for model files and cache

## 2. Webcam and Face Detection
- [ ] 2.1 Implement webcam capture with OpenCV VideoCapture at 1920x1080
- [ ] 2.2 Integrate MediaPipe Face Mesh with tracking mode smoothing enabled
- [ ] 2.3 Add camera selection support via `--camera` argument (default: 0)
- [ ] 2.4 Implement graceful handling when no face detected (show banner, continue)
- [ ] 2.5 Configure MediaPipe: max_num_faces=1, refine_landmarks=True, tracking confidence=0.5

## 3. Eye Region Extraction and Preprocessing
- [ ] 3.1 Extract left eye landmarks using 12-point contour (indices: 362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466)
- [ ] 3.2 Compute bounding box from landmarks and check size >100px (show "Move Closer" if smaller)
- [ ] 3.3 Add 20% padding on all sides
- [ ] 3.4 Implement smart crop: expand bounding box to 1.6:1 aspect ratio before extraction
- [ ] 3.5 Add grayscale conversion
- [ ] 3.6 Implement gamma correction (gamma=0.8) on original crop
- [ ] 3.7 Implement CLAHE (clipLimit=1.5, tileGridSize=8x8) on original crop before resize
- [ ] 3.8 Add resize to 640x400 with INTER_LINEAR interpolation
- [ ] 3.9 Implement normalization (mean=0.5, std=0.5, matching training exactly)

## 4. Model Inference Integration
- [ ] 4.1 Implement ONNX Runtime model loading with strict validation
- [ ] 4.2 Validate model input shape (1,1,400,640) and output shape (1,2,400,640) on startup
- [ ] 4.3 Add required `--model` command-line argument (no default path)
- [ ] 4.4 Add GPU execution provider support (CUDAExecutionProvider preferred)
- [ ] 4.5 Fallback to CPUExecutionProvider with warning message if GPU unavailable
- [ ] 4.6 Log which execution provider is active on startup
- [ ] 4.7 Implement inference with proper input tensor format (1,1,400,640) float32
- [ ] 4.8 Implement post-processing (argmax for segmentation mask)
- [ ] 4.9 Add inference time measurement (milliseconds, 1 decimal place)

## 5. Visualization and Overlay
- [ ] 5.1 Implement coordinate transformation (640x400 model space → eye crop → 1080p frame space)
- [ ] 5.2 Create green segmentation overlay (pupil=green at 50% alpha, background=transparent)
- [ ] 5.3 Keep overlay at model resolution (640x400) and upscale to match eye crop region
- [ ] 5.4 Add semi-transparent overlay blending with cv2.addWeighted
- [ ] 5.5 Draw bounding box around detected eye region (green, 3-pixel thickness)
- [ ] 5.6 Add FPS counter display (top-left, green text, 1.0 font scale)
- [ ] 5.7 Add inference time display in ms (top-left, below FPS, 1 decimal)
- [ ] 5.8 Add GPU/CPU indicator (top-left, below inference time)
- [ ] 5.9 Add top center status banner with semi-transparent black background
- [ ] 5.10 Implement status messages: "Face Detected" (green), "No Face Detected" (yellow), "Move Closer" (yellow)

## 6. User Controls and Configuration
- [ ] 6.1 Implement keyboard controls (ESC to exit, Space to pause/resume)
- [ ] 6.2 Add command-line arguments: --model (required), --camera (default 0), --verbose (optional)
- [ ] 6.3 Implement --verbose flag: log everything (timing, shapes, values, coordinates, timestamps)
- [ ] 6.4 Create full 1920x1080 display window named "VisionAssist Live Demo"
- [ ] 6.5 Implement graceful shutdown and resource cleanup (webcam, ONNX session, windows)

## 7. Documentation
- [ ] 7.1 Write `demo/README.md` with setup instructions for 1080p webcam
- [ ] 7.2 Document command-line arguments: --model (required), --camera, --verbose
- [ ] 7.3 Document keyboard controls: ESC (exit), Space (pause/resume)
- [ ] 7.4 Add troubleshooting section (no camera, low FPS on CPU, import errors, GPU setup)
- [ ] 7.5 Document hardware requirements: 1080p webcam, GPU strongly recommended
- [ ] 7.6 Document optimal conditions: lighting, distance (for >100px eye region), angle
- [ ] 7.7 Add example usage commands and expected output

## 8. Testing and Validation
- [ ] 8.1 Test with trained ONNX model on 1080p laptop webcam
- [ ] 8.2 Verify performance on GPU (target 30+ FPS) and CPU (accept lower FPS)
- [ ] 8.3 Test with different lighting conditions (verify CLAHE and gamma help)
- [ ] 8.4 Test with multiple faces in frame (verify primary face selection)
- [ ] 8.5 Test at various distances (verify "Move Closer" warning when eye <100px)
- [ ] 8.6 Test with no face detected (verify banner shows, no crash, clears overlay)
- [ ] 8.7 Verify preprocessing EXACTLY matches training pipeline from train.py
- [ ] 8.8 Test model validation (try invalid model, verify clear error message)
- [ ] 8.9 Test verbose logging (verify all timing, shapes, values logged correctly)
- [ ] 8.10 Verify green overlay color, 3px bounding box, top center banner positioning
- [ ] 8.11 Test pause/resume functionality (Space key)
- [ ] 8.12 Verify segmentation quality and coordinate mapping accuracy
