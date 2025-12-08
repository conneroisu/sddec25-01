# Webcam Demo Specification

## Purpose

This specification defines the requirements for a real-time webcam demonstration of the VisionAssist pupil segmentation system.

## Requirements

### Requirement: Real-Time Webcam Capture
The system SHALL capture video from a connected 1080p webcam at native resolution (1920x1080) and display in a full HD window for high-quality academic presentations.

#### Scenario: Successful webcam initialization at 1080p
- **WHEN** the application is launched with a valid camera index
- **THEN** the webcam feed SHALL be captured at 1920x1080 resolution
- **AND** displayed in a full 1080p window named "VisionAssist Live Demo"
- **AND** the frame rate SHALL target 30 FPS on GPU-accelerated hardware

#### Scenario: No webcam available
- **WHEN** no webcam is connected or accessible
- **THEN** the application SHALL display an error message
- **AND** exit gracefully with instructions to connect a webcam

#### Scenario: Multiple cameras available
- **WHEN** multiple webcams are connected
- **THEN** the user SHALL be able to select a specific camera via command-line argument
- **AND** the default SHALL be camera index 0

### Requirement: Face and Eye Detection
The system SHALL detect faces in the webcam feed and locate the left eye region using MediaPipe Face Mesh with tracking smoothing for stable landmarks.

#### Scenario: Face detected with visible left eye
- **WHEN** a face is visible in the 1080p webcam frame
- **THEN** MediaPipe Face Mesh SHALL detect the face with tracking mode smoothing enabled
- **AND** extract 12-point left eye contour landmarks (indices: 362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466)
- **AND** the left eye region SHALL be highlighted with a green 3-pixel bounding box overlay

#### Scenario: No face detected
- **WHEN** no face is visible in the webcam frame
- **THEN** the system SHALL display a "No Face Detected" message
- **AND** continue processing subsequent frames without crashing

#### Scenario: Multiple faces in frame
- **WHEN** multiple faces are visible
- **THEN** the system SHALL process the primary (largest or most centered) face
- **AND** ignore other faces to maintain performance

### Requirement: Intelligent Eye Region Cropping
The system SHALL extract and crop the left eye region from detected faces with smart aspect ratio handling and minimum size validation.

#### Scenario: Eye region extracted successfully with smart crop
- **WHEN** left eye landmarks are detected from the 12-point contour
- **THEN** the system SHALL compute a bounding box around the eye
- **AND** verify the region width or height is greater than 100 pixels
- **AND** add 20% padding on all sides
- **AND** expand the bounding box to 1.6:1 aspect ratio (640:400) via smart crop
- **AND** extract the cropped region for preprocessing

#### Scenario: Eye region too small (user too far)
- **WHEN** the computed eye bounding box is less than 100 pixels in width or height
- **THEN** the system SHALL display a "Move Closer" warning in the top center banner
- **AND** skip segmentation for that frame
- **AND** continue processing subsequent frames

#### Scenario: Eye region extends beyond frame boundaries
- **WHEN** the computed eye bounding box (after padding and aspect ratio expansion) extends beyond the frame boundaries
- **THEN** the system SHALL clip the region to frame boundaries
- **AND** handle the edge case gracefully without errors

### Requirement: Preprocessing Pipeline Matching Training
The system SHALL preprocess cropped eye regions using the EXACT same pipeline as training in the correct order to ensure model accuracy.

#### Scenario: Preprocessing applied in correct order
- **WHEN** an eye region is cropped with 1.6:1 aspect ratio
- **THEN** the system SHALL apply grayscale conversion to the original crop
- **AND** apply gamma correction with gamma=0.8 using LUT on the original crop
- **AND** apply CLAHE with clipLimit=1.5 and tileGridSize=(8,8) on the original crop
- **AND** resize to 640x400 pixels using INTER_LINEAR interpolation
- **AND** normalize with mean=0.5 and std=0.5
- **AND** add batch dimension for shape (1, 1, 400, 640)

#### Scenario: Preprocessing parameters exactly match training
- **WHEN** preprocessing is applied
- **THEN** all parameters (gamma=0.8, CLAHE clipLimit=1.5, tileGridSize=8x8, normalization mean=0.5 std=0.5) SHALL match the training pipeline from `train.py` lines 1134-1172
- **AND** the preprocessing order SHALL match training (gamma → CLAHE on original → resize → normalize)
- **AND** the output tensor shape SHALL be (1, 1, 400, 640) with dtype float32

### Requirement: ONNX Model Inference
The system SHALL load the trained ShallowNet ONNX model with strict validation and perform real-time inference using GPU acceleration when available.

#### Scenario: Model loaded and validated successfully
- **WHEN** the application starts with a valid ONNX model path via required `--model` flag
- **THEN** the model SHALL be loaded using ONNX Runtime
- **AND** validate input shape is (1, 1, 400, 640) or (-1, 1, 400, 640)
- **AND** validate output shape is (1, 2, 400, 640) or (-1, 2, 400, 640)
- **AND** try CUDAExecutionProvider first (GPU)
- **AND** fallback to CPUExecutionProvider with warning message if GPU unavailable
- **AND** log which execution provider is active on startup

#### Scenario: Inference performed on eye region with GPU
- **WHEN** a preprocessed eye tensor (1, 1, 400, 640) float32 is ready
- **AND** GPU execution provider is active
- **THEN** the system SHALL run inference and return segmentation logits
- **AND** the output shape SHALL be (1, 2, 400, 640) float32
- **AND** inference time SHALL be measured in milliseconds (1 decimal place)
- **AND** inference time SHALL be displayed below FPS counter

#### Scenario: Model file not found or invalid
- **WHEN** the specified model file does not exist or has incorrect shapes
- **THEN** the application SHALL display a clear error message with the expected path and shapes
- **AND** exit gracefully with instructions to provide a valid ShallowNet ONNX model

### Requirement: Segmentation Visualization Overlay
The system SHALL visualize the pupil segmentation results by overlaying them on the original 1080p webcam feed with green color coding.

#### Scenario: Green segmentation overlay displayed on 1080p frame
- **WHEN** inference produces a segmentation mask (640x400)
- **THEN** the pupil region SHALL be highlighted with a semi-transparent green overlay (50% alpha)
- **AND** the background region SHALL remain transparent
- **AND** the overlay SHALL be kept at 640x400 model resolution and upscaled to match the eye crop region size
- **AND** the overlay SHALL be mapped back to the original 1920x1080 frame coordinates
- **AND** the eye bounding box SHALL be drawn in green with 3-pixel thickness

#### Scenario: Coordinate transformation from model to 1080p frame
- **WHEN** mapping segmentation from model space (640x400) to frame space (1920x1080)
- **THEN** the system SHALL upscale the mask from 640x400 to match the eye crop region dimensions
- **AND** then map the upscaled mask to the correct position in the 1920x1080 frame
- **AND** blend the overlay using cv2.addWeighted with original frame

### Requirement: Performance Monitoring and Display
The system SHALL display real-time performance metrics and status messages for transparency during academic presentations.

#### Scenario: Performance metrics displayed in top-left
- **WHEN** the application is running
- **THEN** the current frames per second SHALL be calculated using a rolling average (last 30 frames)
- **AND** displayed in the top-left corner with green text at 1.0 font scale
- **AND** the inference time in milliseconds (1 decimal place) SHALL be displayed below FPS
- **AND** the execution provider (GPU/CPU) SHALL be displayed below inference time
- **AND** all metrics SHALL update in real-time

#### Scenario: Status banner displayed in top center
- **WHEN** processing frames
- **THEN** a semi-transparent black banner SHALL appear in the top center
- **AND** display "Face Detected" in green when a face is found and processing
- **OR** display "No Face Detected" in yellow when no face is detected
- **OR** display "Move Closer" in yellow when eye region is too small (<100px)

#### Scenario: Honest FPS reporting without frame skipping
- **WHEN** processing cannot maintain 30 FPS (e.g., on CPU-only systems)
- **THEN** the system SHALL process all frames without skipping
- **AND** display the actual FPS achieved (e.g., 15 FPS, 22 FPS)
- **AND** NOT drop frames to artificially maintain 30 FPS

### Requirement: User Controls and Configuration
The system SHALL provide keyboard controls and command-line configuration options optimized for academic presentation use cases.

#### Scenario: Exit application gracefully
- **WHEN** the user presses the ESC key
- **THEN** the application SHALL exit gracefully
- **AND** release all resources (webcam, ONNX session, MediaPipe, windows)

#### Scenario: Pause and resume with spacebar
- **WHEN** the user presses the spacebar
- **THEN** the video feed SHALL pause (freeze current frame)
- **AND** pressing spacebar again SHALL resume processing from live feed

#### Scenario: Required model path argument
- **WHEN** the application is launched
- **THEN** the user MUST specify the ONNX model path via `--model` argument
- **AND** this argument SHALL be required (no default path)
- **AND** the application SHALL exit with error if `--model` is not provided

#### Scenario: Optional camera selection
- **WHEN** the application is launched
- **THEN** the user MAY specify camera index via `--camera` argument
- **AND** the default SHALL be camera index 0 if not specified

#### Scenario: Optional verbose logging
- **WHEN** the application is launched with `--verbose` flag
- **THEN** the system SHALL log comprehensive debugging information including:
  - Frame capture timing
  - Face detection timing and confidence
  - First 3 landmark coordinates (to avoid spam)
  - Preprocessing timing at each step
  - Tensor shapes and min/max values after each transform
  - Inference timing breakdown
  - Coordinate transformation details
- **AND** all logs SHALL be timestamped

### Requirement: Documentation and Setup Instructions
The system SHALL provide comprehensive documentation for setup, usage, and troubleshooting to enable easy demonstration preparation.

#### Scenario: README with setup instructions
- **WHEN** a user reads `demo/README.md`
- **THEN** it SHALL contain installation instructions for all dependencies
- **AND** explain how to obtain or specify the ONNX model
- **AND** provide example commands to run the demo

#### Scenario: README with troubleshooting guide
- **WHEN** a user encounters issues
- **THEN** the README SHALL include common problems (no camera, low FPS, import errors)
- **AND** provide solutions for each problem

#### Scenario: README with hardware requirements
- **WHEN** a user reviews requirements
- **THEN** the README SHALL specify minimum hardware (webcam, CPU/GPU)
- **AND** recommend optimal conditions (lighting, distance, angle)

### Requirement: Error Handling and Graceful Degradation
The system SHALL handle errors gracefully and continue operation when possible to ensure robust demonstration performance.

#### Scenario: Webcam disconnected during operation
- **WHEN** the webcam is disconnected while running
- **THEN** the system SHALL detect the disconnection
- **AND** display an error message
- **AND** attempt to reconnect or exit gracefully

#### Scenario: Inference fails on a frame
- **WHEN** ONNX inference encounters an error on a specific frame
- **THEN** the system SHALL log the error
- **AND** skip that frame and continue processing subsequent frames

#### Scenario: MediaPipe detection fails intermittently
- **WHEN** MediaPipe fails to detect a face on some frames
- **THEN** the system SHALL display "No Face" status
- **AND** continue processing without crashing
- **AND** resume normal operation when face is detected again

