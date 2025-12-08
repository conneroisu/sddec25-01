# Design Document: Live Webcam Demo Application

## Context

The VisionAssist project has successfully trained a ShallowNet semantic segmentation model for eye/pupil detection using the OpenEDS dataset. The model expects 640x400 grayscale eye images and outputs 2-class segmentation (background vs pupil). To demonstrate this capability for academic presentations, we need a real-time webcam application.

**Constraints:**
- Model is already trained (ONNX format)
- Target: 30+ FPS real-time performance
- Use case: Academic presentations, poster sessions
- Input: Standard webcam (full face capture, not close-up eye camera)
- Output: Desktop window with real-time segmentation overlay

**Key Challenge:** The model expects close-up eye images (640x400), but webcam captures full face/scene at higher resolution. We need intelligent face/eye detection and cropping.

## Goals / Non-Goals

**Goals:**
- Real-time webcam demo running at 30+ FPS
- Accurate face and eye detection with robust cropping
- Preprocessing pipeline matching training (CLAHE, gamma correction, normalization)
- Visual segmentation overlay on live video feed
- Simple one-command launch for presentations
- Performance monitoring (FPS, inference time)

**Non-Goals:**
- Web-based interface (desktop only)
- Multi-person tracking (focus on single primary face)
- Recording/saving video streams (optional screenshot only)
- Model training or fine-tuning (inference only)
- Mobile/embedded deployment (desktop only)

## Architecture

### High-Level Pipeline

```
Webcam Frame (1920x1080 RGB)
    ↓
MediaPipe Face Mesh Detection
    ↓
Eye Landmarks Extraction (left/right eye points)
    ↓
Intelligent Eye Region Cropping (with padding)
    ↓
Preprocessing (grayscale → CLAHE → gamma → resize 640x400 → normalize)
    ↓
ONNX Inference (ShallowNet model)
    ↓
Segmentation Output (640x400, 2 classes)
    ↓
Post-processing (argmax → binary mask)
    ↓
Coordinate Transformation (model space → original frame space)
    ↓
Overlay Visualization (semi-transparent colored mask)
    ↓
Display with FPS Counter
```

### Component Breakdown

#### 1. Webcam Capture Module
- **Library:** OpenCV `cv2.VideoCapture`
- **Resolution:** 1920x1080 (Full HD) for high-quality presentations
- **Frame Rate:** 30 FPS target (will process all frames, accept lower FPS if hardware limited)
- **Features:**
  - Multi-camera support (select via `--camera` CLI argument)
  - Graceful error handling for missing camera
  - No frame skipping - process every frame for accuracy
  - Full 1080p display window for impressive visual quality

#### 2. Face Detection Module
- **Library:** MediaPipe Face Mesh
- **Why MediaPipe:**
  - GPU-accelerated, 30+ FPS on modern hardware
  - Provides 468 facial landmarks including detailed eye regions
  - More accurate than Haar cascades, faster than dlib
  - Well-maintained, production-ready
- **Configuration:**
  - `max_num_faces=1` (track primary face only)
  - `refine_landmarks=True` (better eye accuracy)
  - `min_detection_confidence=0.5`
  - `min_tracking_confidence=0.5`
- **Eye Selection:** Left eye only (configurable for future extension)
- **Output:** Eye landmarks (points defining upper/lower eyelid, corners)

#### 3. Eye Region Extraction Module
- **Input:** MediaPipe left eye landmarks (12-point contour)
  - Landmark indices: 362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466
- **Algorithm:**
  1. Compute bounding box from 12 left eye contour landmarks
  2. Check if region > 100px (width or height) - if not, show "Move Closer" warning
  3. Add 20% padding on each side to capture full eye context
  4. Smart crop: Expand bounding box to 1.6:1 aspect ratio (640:400) before extraction
  5. Handle edge cases (crop extends beyond frame - clip to boundaries)
- **Smoothing:** Use MediaPipe's built-in tracking mode smoothing for stable landmarks
- **Output:** Cropped eye region as numpy array with 1.6:1 aspect ratio
- **Eye Selection:** Left eye only (fixed for performance and simplicity)

#### 4. Preprocessing Module
**Critical:** Must match training pipeline exactly from `train.py`

- **Step 1: Grayscale Conversion**
  - `cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)` if RGB input
  - Input: Smart-cropped eye region at original resolution

- **Step 2: Gamma Correction** (on original crop)
  - Gamma value: 0.8 (from training: `self.gamma_table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)`)
  - Apply: `cv2.LUT(image, gamma_table)`

- **Step 3: CLAHE** (on original crop before resize)
  - Parameters: `clipLimit=1.5, tileGridSize=(8, 8)` (from training)
  - Purpose: Enhance local contrast in eye region
  - Applied to original resolution for best quality

- **Step 4: Resize to Model Input**
  - Target: 640x400 (from training: `IMAGE_HEIGHT = 400, IMAGE_WIDTH = 640`)
  - Interpolation: `cv2.INTER_LINEAR`
  - Input already has 1.6:1 aspect ratio from smart crop

- **Step 5: Normalization**
  - Convert to tensor, normalize with mean=0.5, std=0.5
  - `transforms.Normalize([0.5], [0.5])` (from training)
  - Result: Values in range [-1, 1]

- **Step 6: Add Batch Dimension**
  - Shape: (1, 1, 400, 640) for ONNX input

**Verbose Logging:** If --verbose flag set, log tensor shapes, min/max values, and timing at each step

#### 5. Inference Module
- **Runtime:** ONNX Runtime
- **Execution Providers:**
  - Try GPU first: `CUDAExecutionProvider` (preferred for 1080p processing)
  - Fallback to CPU: `CPUExecutionProvider` with warning message
  - Log which provider is active on startup
- **Model Validation on Startup:**
  - Check input shape is (1, 1, 400, 640) or (-1, 1, 400, 640)
  - Check output shape is (1, 2, 400, 640) or (-1, 2, 400, 640)
  - Exit with clear error if shapes don't match
- **Input:** Tensor shape (1, 1, 400, 640), dtype float32
- **Output:** Tensor shape (1, 2, 400, 640), dtype float32 (logits for 2 classes)
- **Post-processing:** `np.argmax(output, axis=1)` → binary mask (0=background, 1=pupil)
- **Performance:** Accept actual inference time, no frame dropping - report honest metrics

#### 6. Visualization Module
- **Coordinate Transformation:**
  - Segmentation mask is 640x400 (model space)
  - Keep at model resolution (don't upscale to full 1080p)
  - Map to eye crop region coordinates
  - Then map to original 1920x1080 frame coordinates
- **Overlay Creation:**
  - Color map: Background (transparent), Pupil (green with 50% alpha)
  - Upscale 640x400 mask to match eye crop region size (bilinear interpolation)
  - Blend with original 1080p frame using `cv2.addWeighted`
- **Additional Overlays:**
  - Draw bounding box around detected eye region (green, 3-pixel thickness)
  - FPS counter (top-left, green text, 1.0 font scale)
  - Inference time in ms (top-left, below FPS)
  - GPU/CPU indicator (top-left, below inference time)
- **Status Messages (Top Center Banner):**
  - "Face Detected" (green) when processing normally
  - "No Face Detected" (yellow) when no face found
  - "Move Closer" (yellow) when eye region < 100px
  - Banner: Semi-transparent black background, white/colored text
- **Fallback Behavior:**
  - When no face detected: Clear overlay, show "No Face Detected" banner
  - Do NOT show stale segmentation from previous frames
  - When region too small: Show "Move Closer" banner, no segmentation

#### 7. Display and Control Module
- **Window:** `cv2.imshow` with named window "VisionAssist Live Demo"
  - Full 1920x1080 resolution display
  - Single view (no debug panels)
- **Keyboard Controls:**
  - ESC: Exit application gracefully
  - Space: Pause/resume (freeze current frame, keep processing on resume)
- **Performance Monitoring:**
  - FPS calculation using rolling average (last 30 frames)
  - Inference time per frame (milliseconds, 1 decimal place)
  - Total pipeline time per frame (capture → display)
  - Display actual FPS (no frame skipping even if below 30)
  - Show GPU/CPU execution provider being used
- **Logging:**
  - Minimal output by default (startup info, errors, warnings only)
  - `--verbose` flag for EVERYTHING:
    - Frame capture timing
    - Face detection timing + confidence
    - Landmark coordinates (first 3 points only to avoid spam)
    - Preprocessing timing at each step
    - Tensor shapes and min/max values after each transform
    - Inference timing breakdown
    - Coordinate transformation details
  - All verbose logs timestamped

## Decisions

### Decision 1: MediaPipe Face Mesh vs Alternatives

**Options Considered:**
1. **MediaPipe Face Mesh** (chosen)
   - Pros: Fast (30+ FPS), accurate landmarks, GPU accelerated, maintained
   - Cons: Larger dependency, requires MediaPipe installation
2. **dlib Face Detector + Shape Predictor**
   - Pros: Very accurate, well-established
   - Cons: Slower (~10 FPS), CPU-bound, larger model files
3. **OpenCV Haar Cascades**
   - Pros: Lightweight, fast, built into OpenCV
   - Cons: Less accurate for eye landmarks, requires manual eye region estimation

**Decision:** MediaPipe for real-time performance and accuracy requirements.

### Decision 2: Single Eye vs Both Eyes

**Options Considered:**
1. **Left eye only** (chosen)
   - Pros: Half the inference cost, simpler visualization, cleaner demo
   - Cons: Less comprehensive demo
2. **Both eyes independently**
   - Pros: More complete demo, shows robustness
   - Cons: 2x inference cost, may drop below 30 FPS target
3. **Alternating eyes per frame**
   - Pros: Performance optimization with both eyes shown
   - Cons: Flickering visualization, confusing UX

**Decision:** Left eye only (fixed). This simplifies the implementation and ensures consistent real-time performance even on CPU-only systems.

### Decision 3: Desktop Window vs Web Interface

**Options Considered:**
1. **Desktop window (OpenCV)** (chosen)
   - Pros: Low latency, simple setup, no network overhead, better for presentations
   - Cons: Less shareable, platform-specific
2. **Web interface (Flask/FastAPI)**
   - Pros: Accessible from any device, shareable
   - Cons: Added complexity, network latency, harder deployment for presentations

**Decision:** Desktop for academic presentation use case (easy setup, reliable performance).

### Decision 4: ONNX Runtime vs PyTorch Inference

**Options Considered:**
1. **ONNX Runtime** (chosen)
   - Pros: Optimized for inference, cross-platform, lighter dependencies
   - Cons: Model must be exported (already done in training)
2. **PyTorch inference**
   - Pros: Native support, easier debugging
   - Cons: Heavier dependencies, slower inference, requires model code

**Decision:** ONNX Runtime for production-ready inference performance.

## Data Flow

```python
# Pseudocode for main loop

while True:
    # 1. Capture frame
    ret, frame = cap.read()  # e.g., (720, 1280, 3)

    # 2. Face detection
    results = face_mesh.process(frame)
    if not results.multi_face_landmarks:
        display_no_face_message()
        continue

    # 3. Extract eye landmarks
    landmarks = results.multi_face_landmarks[0]
    eye_points = extract_eye_landmarks(landmarks)  # 6 points for left eye

    # 4. Crop eye region
    x_min, y_min, x_max, y_max = compute_eye_bbox(eye_points, padding=0.2)
    eye_crop = frame[y_min:y_max, x_min:x_max]  # e.g., (120, 192, 3)

    # 5. Preprocess
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    gamma = cv2.LUT(gray, gamma_table)
    clahe_img = clahe.apply(gamma)
    resized = cv2.resize(clahe_img, (640, 400))
    normalized = (resized / 255.0 - 0.5) / 0.5  # mean=0.5, std=0.5
    input_tensor = normalized[np.newaxis, np.newaxis, ...]  # (1, 1, 400, 640)

    # 6. Inference
    output = onnx_session.run(None, {'input': input_tensor.astype(np.float32)})[0]
    # output shape: (1, 2, 400, 640)

    # 7. Post-process
    mask = np.argmax(output[0], axis=0)  # (400, 640), values 0 or 1

    # 8. Visualization
    mask_resized = cv2.resize(mask, (x_max - x_min, y_max - y_min))
    overlay = create_colored_overlay(mask_resized)  # red for pupil
    frame[y_min:y_max, x_min:x_max] = cv2.addWeighted(
        frame[y_min:y_max, x_min:x_max], 0.7, overlay, 0.3, 0
    )

    # 9. Add UI elements
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), ...)

    # 10. Display
    cv2.imshow('VisionAssist Demo', frame)

    # 11. Handle keyboard
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
```

## Risks / Trade-offs

### Risk 1: Real-Time Performance on CPU-Only Systems
- **Risk:** May not achieve 30 FPS on older CPUs without GPU
- **Mitigation:**
  - Use ONNX Runtime optimizations
  - Default to 640x480 webcam resolution for better performance
  - Process all frames (no skipping) but report actual FPS honestly
  - Provide performance tuning guide in README
  - User acceptance: Demo prioritizes accuracy over strict 30 FPS requirement

### Risk 2: Preprocessing Mismatch with Training
- **Risk:** If preprocessing doesn't exactly match training, segmentation quality degrades
- **Mitigation:**
  - Copy exact preprocessing code from `train.py`
  - Document preprocessing steps explicitly
  - Add validation script to compare preprocessing outputs
  - Test with sample images from OpenEDS first

### Risk 3: Eye Detection Failures
- **Risk:** MediaPipe may fail with glasses, poor lighting, extreme angles
- **Mitigation:**
  - Graceful degradation (show message, continue waiting for face)
  - Add confidence threshold adjustments
  - Document optimal conditions in README (lighting, distance, angle)

### Risk 4: Model Domain Mismatch
- **Risk:** Model trained on VR headset eyes may not generalize to webcam captures
- **Mitigation:**
  - This is expected (document in README as known limitation)
  - Still demonstrates the technology and pipeline
  - Future work: fine-tune on webcam data or use transfer learning

## Migration Plan

N/A - New capability, no migration needed.

## Design Decisions (Finalized)

### Core Configuration
1. **Eye Selection:** Left eye only - simpler, faster, half the inference cost
2. **Webcam Resolution:** 1920x1080 (Full HD) native capture - high quality for presentations
3. **Display Resolution:** Full 1080p window - impressive presentation quality
4. **Model File:** User provides via required `--model` flag - keeps repo lightweight, explicit
5. **Fallback Mode:** Clear overlay and show "No Face Detected" message - honest and transparent

### Eye Detection & Cropping
6. **Landmark Points:** Full 12-point eye contour (MediaPipe indices: 362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466)
7. **Crop Padding:** 20% padding around eye landmarks - balanced, captures eye plus surrounding context
8. **Aspect Ratio Handling:** Smart crop - expand bounding box to 1.6:1 (640:400) before extraction
9. **Minimum Size Check:** Eye region must be > 100px - show "Move Closer" warning if smaller
10. **Landmark Smoothing:** Use MediaPipe built-in tracking mode smoothing - reduces jitter

### Preprocessing Pipeline
11. **Preprocessing Order:** Gamma → CLAHE on original crop → Resize to 640x400 → Normalize
12. **Gamma Correction:** Fixed γ=0.8 matching training - most accurate, no runtime complexity
13. **CLAHE Timing:** Apply to original crop before resize - better quality, matches training

### Visualization
14. **Overlay Color:** Green (50% alpha) - tech aesthetic, complements skin tones
15. **Overlay Resolution:** Match model output (640x400 upscaled) - shows direct model prediction
16. **Bounding Box:** 3-pixel green line - bold and clear for presentations
17. **Font Size:** Medium (1.0 scale) - readable at desk distance
18. **Message Position:** Top center banner - clear without blocking main view
19. **Debug View:** No separate debug panels - clean single-window interface

### Performance & Error Handling
20. **Frame Processing:** Process every frame, accept actual FPS - honest performance reporting
21. **GPU Strategy:** Prefer GPU (CUDA), fallback to CPU with warning - optimized for 1080p
22. **Slow Inference:** Accept and show actual FPS/timing - simple, transparent
23. **Model Validation:** Strict startup validation of input/output shapes - fail fast with clear errors
24. **Logging:** Minimal by default, `--verbose` logs everything (timing, shapes, values, coordinates)

### User Experience
25. **Capture Feature:** No screenshot/recording - keep it simple
26. **Model Path:** Always require explicit `--model` flag - clear and unambiguous
27. **Controls:** ESC (exit), Space (pause/resume) - simple and standard
