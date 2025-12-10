#!/usr/bin/env python3
"""
VisionAssist Live Demo - Dual Eye Segmentation with NSA Model

This demo performs real-time semantic segmentation on BOTH eyes using webcam input.
It uses the NSA (Native Sparse Attention) model for pupil segmentation.

The application captures live video, runs inference on both left and right eyes,
and visualizes the segmentation results with dual overlays.
"""

import argparse
import platform
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from nsa import create_nsa_pupil_seg


class VisionAssistDemoDD:
    """Dual-eye demo application for VisionAssist live webcam inference with NSA model."""

    # MediaPipe left eye landmark indices (12 points around the eye)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466]

    # MediaPipe right eye landmark indices (12 points around the eye)
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144, 145, 163, 7, 154, 155, 246]

    # Target aspect ratio for eye region (width:height = 640:400 = 1.6:1)
    TARGET_ASPECT_RATIO = 640 / 400

    # Normalization parameters (no preprocessing - just normalize)
    NORMALIZE_MEAN = 0.5
    NORMALIZE_STD = 0.5

    # Model input/output dimensions
    MODEL_WIDTH = 640
    MODEL_HEIGHT = 400

    # Display settings
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    MIN_EYE_REGION_SIZE = 100  # Minimum bounding box size
    BBOX_PADDING = 0.2  # 20% padding on each side
    OVERLAY_ALPHA = 0.5

    def __init__(
        self,
        model_path: str,
        model_size: str = "small",
        camera_index: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize the VisionAssist dual-eye demo with NSA model.

        Args:
            model_path: Path to PyTorch model checkpoint (.pt or .pth file)
            model_size: NSA model size ('pico', 'nano', 'tiny', 'small', 'medium')
            camera_index: Camera device index (default 0)
            verbose: Enable comprehensive logging
        """
        self.model_path = model_path
        self.model_size = model_size
        self.camera_index = camera_index
        self.verbose = verbose

        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_frame_time = time.time()

        # Initialize components
        self._init_camera()
        self._init_model()
        self._init_face_mesh()

        # State
        self.paused = False
        self.frame_count = 0

        # Pre-allocated buffers for preprocessing (avoid per-frame allocations)
        # Eye crop resized to model size (BGR for initial resize)
        self._eye_crop_resized = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH, 3), dtype=np.uint8
        )
        # Grayscale conversion buffer
        self._gray_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        self._normalized_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.float32
        )
        self._input_tensor = np.empty(
            (1, 1, self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.float32
        )

        # Pre-allocated buffer for inference output (one per eye)
        self._mask_buffer_left = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        self._mask_buffer_right = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        # Pre-allocated buffer for argmax output - int64 for np.argmax out=
        self._argmax_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.int64
        )

        # Pre-allocated buffer for eye extraction (12 landmark points)
        self._eye_points_buffer = np.empty(
            (max(len(self.LEFT_EYE_INDICES), len(self.RIGHT_EYE_INDICES)), 2),
            dtype=np.int32,
        )

        # Pre-allocated buffers for visualization (sized lazily on first frame)
        self._frame_rgb = None
        self._overlay_buffer = None
        self._green_overlay_cache_left = None
        self._green_overlay_cache_right = None
        self._green_overlay_size_left = (0, 0)
        self._green_overlay_size_right = (0, 0)
        # Pre-allocated buffer for visualization mask resize (max camera resolution)
        self._mask_viz_buffer = np.empty(
            (self.CAMERA_HEIGHT, self.CAMERA_WIDTH), dtype=np.uint8
        )

    def _init_camera(self):
        """Initialize webcam capture."""
        if self.verbose:
            print(f"Initializing camera {self.camera_index}...")

        # On macOS, use AVFoundation backend for proper webcam access
        if platform.system() == "Darwin":
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera {self.camera_index}. "
                "Check that the camera is connected and not in use."
            )

        # Set resolution to Full HD
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.verbose:
            print(
                f"Camera initialized: {actual_width}x{actual_height} "
                f"(requested {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT})"
            )

    def _init_model(self):
        """Initialize NSA PyTorch model."""
        if self.verbose:
            print(f"Loading NSA model (size={self.model_size}) from {self.model_path}...")

        # Determine device (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.execution_provider = "CUDAExecutionProvider"
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.execution_provider = "MPSExecutionProvider"
        else:
            self.device = torch.device("cpu")
            self.execution_provider = "CPUExecutionProvider"

        if self.verbose:
            print(f"Using device: {self.device}")

        # Create NSA model architecture
        self.model = create_nsa_pupil_seg(
            size=self.model_size,
            in_channels=1,
            num_classes=2,
        )

        # Load checkpoint weights
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"NSA Model loaded with {num_params:,} parameters")
            print(f"Model input: (B, 1, H, W), output: (B, 2, H, W)")

    def _init_face_mesh(self):
        """Initialize MediaPipe Face Mesh."""
        if self.verbose:
            print("Initializing MediaPipe Face Mesh...")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        if self.verbose:
            print("MediaPipe Face Mesh initialized")

    def _extract_eye_region(self, frame, landmarks, eye_indices):
        """
        Extract eye region from frame using MediaPipe landmarks.

        Args:
            frame: Input BGR frame
            landmarks: MediaPipe face landmarks
            eye_indices: List of landmark indices for the eye

        Returns:
            tuple: (eye_crop, bbox) where bbox is (x, y, w, h), or (None, None)
        """
        h, w = frame.shape[:2]

        # Extract eye landmark coordinates into pre-allocated buffer
        for i, idx in enumerate(eye_indices):
            landmark = landmarks.landmark[idx]
            self._eye_points_buffer[i, 0] = int(landmark.x * w)
            self._eye_points_buffer[i, 1] = int(landmark.y * h)

        # Use only the portion of the buffer that was filled
        eye_points = self._eye_points_buffer[: len(eye_indices)]

        if self.verbose:
            print(f"  First 3 eye landmarks: {eye_points[:3].tolist()}")

        # Compute bounding box
        x_min, y_min = eye_points.min(axis=0)
        x_max, y_max = eye_points.max(axis=0)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # Check if eye region is large enough
        if bbox_w < self.MIN_EYE_REGION_SIZE or bbox_h < self.MIN_EYE_REGION_SIZE:
            return None, None

        # Add padding (20% on each side)
        pad_w = int(bbox_w * self.BBOX_PADDING)
        pad_h = int(bbox_h * self.BBOX_PADDING)

        x_min = max(0, x_min - pad_w)
        y_min = max(0, y_min - pad_h)
        x_max = min(w, x_max + pad_w)
        y_max = min(h, y_max + pad_h)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # Expand to 1.6:1 aspect ratio (640:400)
        current_ratio = bbox_w / bbox_h
        if current_ratio < self.TARGET_ASPECT_RATIO:
            # Too narrow, expand width
            target_w = int(bbox_h * self.TARGET_ASPECT_RATIO)
            diff = target_w - bbox_w
            x_min = max(0, x_min - diff // 2)
            x_max = min(w, x_max + diff // 2)
            bbox_w = x_max - x_min
        else:
            # Too short, expand height
            target_h = int(bbox_w / self.TARGET_ASPECT_RATIO)
            diff = target_h - bbox_h
            y_min = max(0, y_min - diff // 2)
            y_max = min(h, y_max + diff // 2)
            bbox_h = y_max - y_min

        # Extract region
        eye_crop = frame[y_min:y_max, x_min:x_max]

        # Validate the crop is not empty
        if eye_crop.size == 0:
            return None, None

        return eye_crop, (x_min, y_min, bbox_w, bbox_h)

    def _preprocess(self, eye_crop):
        """
        Preprocess eye region for model inference.
        NO PREPROCESSING - just resize, grayscale, and normalize.

        Uses pre-allocated buffers to avoid per-frame memory allocations.

        Args:
            eye_crop: BGR image of eye region

        Returns:
            np.ndarray: Preprocessed tensor of shape (1, 1, 400, 640)
        """
        if self.verbose:
            t_start = time.time()

        # Step 1: Resize to model input size
        cv2.resize(
            eye_crop,
            (self.MODEL_WIDTH, self.MODEL_HEIGHT),
            dst=self._eye_crop_resized,
            interpolation=cv2.INTER_LINEAR,
        )
        if self.verbose:
            t_resize = time.time()
            print(
                f"  Resize: {(t_resize - t_start)*1000:.2f}ms, "
                f"shape={self._eye_crop_resized.shape}"
            )

        # Step 2: Convert to grayscale
        cv2.cvtColor(self._eye_crop_resized, cv2.COLOR_BGR2GRAY, dst=self._gray_buffer)
        if self.verbose:
            t_gray = time.time()
            print(
                f"  Grayscale: {(t_gray - t_resize)*1000:.2f}ms, "
                f"shape={self._gray_buffer.shape}, range=[{self._gray_buffer.min()}, {self._gray_buffer.max()}]"
            )

        # Step 3: Normalize (mean=0.5, std=0.5) -> range [-1, 1]
        # NO CLAHE, NO GAMMA - just normalize directly
        np.multiply(self._gray_buffer, 1.0 / 255.0, out=self._normalized_buffer)
        np.subtract(
            self._normalized_buffer, self.NORMALIZE_MEAN, out=self._normalized_buffer
        )
        np.divide(
            self._normalized_buffer, self.NORMALIZE_STD, out=self._normalized_buffer
        )
        if self.verbose:
            t_normalize = time.time()
            print(
                f"  Normalize: {(t_normalize - t_gray)*1000:.2f}ms, "
                f"range=[{self._normalized_buffer.min():.3f}, {self._normalized_buffer.max():.3f}]"
            )

        # Step 4: Fill pre-allocated tensor
        # NSA model expects (B, C, H, W) = (1, 1, 400, 640)
        self._input_tensor[0, 0] = self._normalized_buffer

        if self.verbose:
            t_end = time.time()
            print(
                f"  Final tensor shape: {self._input_tensor.shape}, "
                f"total preprocessing: {(t_end - t_start)*1000:.2f}ms"
            )

        return self._input_tensor

    def _run_inference(self, input_tensor, mask_buffer):
        """
        Run model inference on preprocessed input.

        Uses pre-allocated mask buffer to avoid per-frame allocations.

        Args:
            input_tensor: Preprocessed tensor of shape (1, 1, 400, 640)
            mask_buffer: Pre-allocated buffer to store the output mask

        Returns:
            tuple: (mask, inference_time) where mask is shape (400, 640)
        """
        t_start = time.time()

        # Convert numpy array to torch tensor
        input_torch = torch.from_numpy(input_tensor).to(self.device)

        # Run inference with no gradient computation
        with torch.no_grad():
            output = self.model(input_torch)

        # Convert output to numpy for post-processing
        output_np = output.cpu().numpy()

        # Post-processing: argmax to get binary mask
        # Model outputs (B, C, H, W) = (1, 2, 400, 640), argmax over classes gives (400, 640)
        np.argmax(output_np[0], axis=0, out=self._argmax_buffer)
        # Copy to mask buffer (auto-casts int64/int32 to uint8)
        mask_buffer[:] = self._argmax_buffer

        inference_time = (time.time() - t_start) * 1000  # Convert to ms

        if self.verbose:
            print(
                f"  Inference: {inference_time:.1f}ms, "
                f"output shape={output_np.shape}, mask shape={mask_buffer.shape}, "
                f"mask values=[{mask_buffer.min()}, {mask_buffer.max()}]"
            )

        return mask_buffer, inference_time

    def _visualize(
        self,
        frame,
        left_eye_crop,
        left_mask,
        left_bbox,
        right_eye_crop,
        right_mask,
        right_bbox,
        inference_time,
        face_detected,
    ):
        """
        Visualize segmentation results on frame with dual overlays.

        Args:
            frame: Original BGR frame
            left_eye_crop: Left eye region crop
            left_mask: Binary segmentation mask for left eye (400, 640)
            left_bbox: Left eye bounding box (x, y, w, h)
            right_eye_crop: Right eye region crop
            right_mask: Binary segmentation mask for right eye (400, 640)
            right_bbox: Right eye bounding box (x, y, w, h)
            inference_time: Total inference time in milliseconds
            face_detected: Whether face was detected

        Returns:
            np.ndarray: Annotated frame
        """
        # Reuse overlay buffer if same size, otherwise reallocate
        if self._overlay_buffer is None or self._overlay_buffer.shape != frame.shape:
            self._overlay_buffer = np.empty_like(frame)

        np.copyto(self._overlay_buffer, frame)
        annotated = self._overlay_buffer

        # Draw status banner at top center
        banner_height = 50
        banner_y = 0
        banner_x = 0
        banner_w = annotated.shape[1]

        # Semi-transparent black background
        banner_region = annotated[
            banner_y : banner_y + banner_height, banner_x : banner_x + banner_w
        ]
        np.multiply(banner_region, 0.5, out=banner_region, casting="unsafe")

        # Status text
        if not face_detected:
            status_text = "No Face Detected"
            status_color = (0, 255, 255)  # Yellow
        elif left_mask is None and right_mask is None:
            status_text = "Move Closer"
            status_color = (0, 255, 255)  # Yellow
        else:
            eyes_detected = sum([left_mask is not None, right_mask is not None])
            status_text = f"Both Eyes Detected" if eyes_detected == 2 else f"{eyes_detected} Eye Detected"
            status_color = (0, 255, 0)  # Green

        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (banner_w - text_size[0]) // 2
        text_y = banner_y + (banner_height + text_size[1]) // 2
        cv2.putText(
            annotated,
            status_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2,
        )

        # Overlay LEFT eye mask (green)
        if left_mask is not None and left_bbox is not None:
            x, y, w, h = left_bbox

            # Resize mask to match eye crop size
            mask_view = self._mask_viz_buffer[:h, :w]
            cv2.resize(left_mask, (w, h), dst=mask_view, interpolation=cv2.INTER_NEAREST)

            # Reuse green overlay cache if same size
            if self._green_overlay_size_left != (h, w):
                self._green_overlay_cache_left = np.zeros((h, w, 3), dtype=np.uint8)
                self._green_overlay_size_left = (h, w)
            else:
                self._green_overlay_cache_left.fill(0)

            self._green_overlay_cache_left[mask_view == 1] = (0, 255, 0)

            # Blend with original eye region
            eye_region = annotated[y : y + h, x : x + w]
            cv2.addWeighted(
                eye_region,
                1 - self.OVERLAY_ALPHA,
                self._green_overlay_cache_left,
                self.OVERLAY_ALPHA,
                0,
                eye_region,
            )

            # Draw bounding box (green for left eye)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(
                annotated,
                "L",
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

        # Overlay RIGHT eye mask (cyan)
        if right_mask is not None and right_bbox is not None:
            x, y, w, h = right_bbox

            # Resize mask to match eye crop size
            mask_view = self._mask_viz_buffer[:h, :w]
            cv2.resize(right_mask, (w, h), dst=mask_view, interpolation=cv2.INTER_NEAREST)

            # Reuse overlay cache if same size
            if self._green_overlay_size_right != (h, w):
                self._green_overlay_cache_right = np.zeros((h, w, 3), dtype=np.uint8)
                self._green_overlay_size_right = (h, w)
            else:
                self._green_overlay_cache_right.fill(0)

            self._green_overlay_cache_right[mask_view == 1] = (255, 255, 0)  # Cyan (BGR)

            # Blend with original eye region
            eye_region = annotated[y : y + h, x : x + w]
            cv2.addWeighted(
                eye_region,
                1 - self.OVERLAY_ALPHA,
                self._green_overlay_cache_right,
                self.OVERLAY_ALPHA,
                0,
                eye_region,
            )

            # Draw bounding box (cyan for right eye)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(
                annotated,
                "R",
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )

        # Draw FPS counter (top-left, below banner)
        fps = self._calculate_fps()
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, banner_height + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # Draw inference time (below FPS)
        if inference_time is not None:
            cv2.putText(
                annotated,
                f"Inference: {inference_time:.1f}ms",
                (10, banner_height + 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

        # Draw execution provider (below inference time)
        provider_short = "GPU" if self.device.type in ("cuda", "mps") else "CPU"
        cv2.putText(
            annotated,
            f"Device: {provider_short}",
            (10, banner_height + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # Draw mode indicator
        cv2.putText(
            annotated,
            f"Mode: NSA Dual Eye ({self.model_size})",
            (10, banner_height + 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
        )

        return annotated

    def _calculate_fps(self):
        """Calculate rolling average FPS."""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        self.last_frame_time = current_time

        if delta > 0:
            self.fps_buffer.append(1.0 / delta)

        if len(self.fps_buffer) > 0:
            return sum(self.fps_buffer) / len(self.fps_buffer)
        return 0.0

    def run(self):
        """Main processing loop."""
        print("\n" + "=" * 80)
        print("VisionAssist Live Demo - NSA DUAL EYE")
        print("=" * 80)
        print(f"Model: {self.model_path}")
        print(f"Model Size: {self.model_size}")
        print(f"Camera: {self.camera_index}")
        print(f"Execution Provider: {self.execution_provider}")
        print(f"Mode: Both eyes with NSA (Native Sparse Attention)")
        print("\nControls:")
        print("  ESC - Exit")
        print("  SPACE - Pause/Resume")
        print("=" * 80 + "\n")

        # Create named window for proper display on macOS
        cv2.namedWindow("VisionAssist NSA Dual Eye Demo", cv2.WINDOW_NORMAL)

        # Allow camera to warm up (important for macOS)
        print("Warming up camera...")
        for i in range(30):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Check if frame is not black (has actual content)
                if frame.mean() > 1.0:
                    print(f"Camera ready after {i+1} warmup frames.\n")
                    break
            time.sleep(0.1)
        else:
            print("WARNING: Camera may not be capturing properly.")
            print("Check System Settings -> Privacy & Security -> Camera permissions.\n")

        try:
            while True:
                if not self.paused:
                    # Capture frame
                    if self.verbose:
                        t_capture = time.time()
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        continue

                    if self.verbose:
                        print(
                            f"\nFrame {self.frame_count}: "
                            f"capture {(time.time() - t_capture)*1000:.2f}ms"
                        )

                    # Convert to RGB for MediaPipe using pre-allocated buffer
                    if self._frame_rgb is None or self._frame_rgb.shape != frame.shape:
                        self._frame_rgb = np.empty_like(frame)
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb)

                    # Face detection
                    if self.verbose:
                        t_face = time.time()
                    results = self.face_mesh.process(self._frame_rgb)
                    face_detected = results.multi_face_landmarks is not None

                    if self.verbose:
                        t_face_end = time.time()
                        print(
                            f"  Face detection: {(t_face_end - t_face)*1000:.2f}ms, "
                            f"detected={face_detected}"
                        )

                    # Initialize variables for both eyes
                    left_eye_crop = None
                    left_bbox = None
                    left_mask = None
                    right_eye_crop = None
                    right_bbox = None
                    right_mask = None
                    total_inference_time = 0

                    # Process if face detected
                    if face_detected:
                        landmarks = results.multi_face_landmarks[0]

                        # Extract LEFT eye region
                        left_eye_crop, left_bbox = self._extract_eye_region(
                            frame, landmarks, self.LEFT_EYE_INDICES
                        )

                        if left_eye_crop is not None:
                            # Preprocess
                            input_tensor = self._preprocess(left_eye_crop)
                            # Run inference
                            left_mask, left_time = self._run_inference(
                                input_tensor, self._mask_buffer_left
                            )
                            total_inference_time += left_time

                        # Extract RIGHT eye region
                        right_eye_crop, right_bbox = self._extract_eye_region(
                            frame, landmarks, self.RIGHT_EYE_INDICES
                        )

                        if right_eye_crop is not None:
                            # Preprocess
                            input_tensor = self._preprocess(right_eye_crop)
                            # Run inference
                            right_mask, right_time = self._run_inference(
                                input_tensor, self._mask_buffer_right
                            )
                            total_inference_time += right_time

                    # Visualize with dual overlays
                    annotated = self._visualize(
                        frame,
                        left_eye_crop,
                        left_mask,
                        left_bbox,
                        right_eye_crop,
                        right_mask,
                        right_bbox,
                        total_inference_time if total_inference_time > 0 else None,
                        face_detected,
                    )

                    # Display
                    cv2.imshow("VisionAssist NSA Dual Eye Demo", annotated)

                    self.frame_count += 1
                else:
                    # Paused - just wait
                    cv2.waitKey(100)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                elif key == ord(" "):  # SPACE
                    self.paused = not self.paused
                    status = "Paused" if self.paused else "Resumed"
                    print(f"\n{status}")

        finally:
            self._cleanup()

    def _cleanup(self):
        """Release resources."""
        if self.verbose:
            print("\nCleaning up resources...")

        if hasattr(self, "cap"):
            self.cap.release()
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()
        cv2.destroyAllWindows()

        if self.verbose:
            print("Cleanup complete")


def main():
    """Main entry point for the VisionAssist NSA dual-eye demo."""
    parser = argparse.ArgumentParser(
        description="VisionAssist Live Demo - NSA Dual Eye Segmentation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch NSA model checkpoint file (REQUIRED)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["pico", "nano", "tiny", "small", "medium"],
        help="NSA model size (default: small)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable comprehensive logging",
    )

    args = parser.parse_args()

    # Initialize and run demo
    demo = VisionAssistDemoDD(
        model_path=args.model,
        model_size=args.model_size,
        camera_index=args.camera,
        verbose=args.verbose,
    )
    demo.run()


if __name__ == "__main__":
    main()
