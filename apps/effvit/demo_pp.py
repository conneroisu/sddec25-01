#!/usr/bin/env python3
"""
TinyEfficientViT Semantic Segmentation with Preprocessing (Gamma + CLAHE)

This demo application performs real-time semantic segmentation on webcam input
using the TinyEfficientViT model. It demonstrates eye tracking and facial feature
detection capabilities for the VisionAssist medical assistive technology project.

The application captures live video, runs inference using PyTorch,
and visualizes the segmentation results with overlays.
"""

import argparse
import platform
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from model import TinyEfficientViTSeg


class VisionAssistDemo:
    """Main demo application for VisionAssist live webcam inference."""

    # MediaPipe left eye landmark indices (12 points around the eye)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466]

    # Target aspect ratio for eye region (width:height = 640:400 = 1.6:1)
    TARGET_ASPECT_RATIO = 640 / 400

    # Preprocessing parameters (MUST match training exactly)
    GAMMA = 0.8
    CLAHE_CLIP_LIMIT = 1.5
    CLAHE_TILE_SIZE = (8, 8)
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

    def __init__(self, model_path: str, camera_index: int = 0, verbose: bool = False):
        """
        Initialize the VisionAssist demo.

        Args:
            model_path: Path to PyTorch model checkpoint (.pt or .pth file)
            camera_index: Camera device index (default 0)
            verbose: Enable comprehensive logging
        """
        self.model_path = model_path
        self.camera_index = camera_index
        self.verbose = verbose

        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_frame_time = time.time()

        # Initialize components
        self._init_camera()
        self._init_model()
        self._init_face_mesh()
        self._init_preprocessing()

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
        # # Gamma correction buffer
        self._gamma_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        # CLAHE buffer (reusing _resized_buffer name for backward compat in normalize step)
        self._resized_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        self._normalized_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.float32
        )
        self._input_tensor = np.empty(
            (1, 1, self.MODEL_WIDTH, self.MODEL_HEIGHT), dtype=np.float32
        )

        # Pre-allocated buffer for inference output
        self._mask_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        # Pre-allocated buffer for argmax output (before transpose) - int64 for np.argmax out=
        self._argmax_buffer = np.empty(
            (self.MODEL_WIDTH, self.MODEL_HEIGHT), dtype=np.int64
        )

        # Pre-allocated buffer for eye extraction (12 landmark points)
        self._eye_points_buffer = np.empty(
            (len(self.LEFT_EYE_INDICES), 2), dtype=np.int32
        )

        # Pre-allocated buffers for visualization (sized lazily on first frame)
        self._frame_rgb = None
        self._overlay_buffer = None
        self._green_overlay_cache = None
        self._green_overlay_size = (0, 0)
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
        """Initialize PyTorch model."""
        if self.verbose:
            print(f"Loading PyTorch model from {self.model_path}...")

        # Determine device (CUDA if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.execution_provider = (
            "CUDAExecutionProvider"
            if self.device.type == "cuda"
            else "CPUExecutionProvider"
        )

        if self.verbose:
            print(f"Using device: {self.device}")

        # Create model architecture
        self.model = TinyEfficientViTSeg(
            in_channels=1,
            num_classes=2,
            embed_dims=(8, 16, 24),
            depths=(1, 1, 1),
            num_heads=(1, 1, 2),
            key_dims=(4, 4, 4),
            attn_ratios=(2, 2, 2),
            window_sizes=(7, 7, 7),
            mlp_ratios=(2, 2, 2),
            decoder_dim=16,
        )

        # Load checkpoint weights
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=True
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
            print(f"Model loaded with {num_params:,} parameters")
            print(f"Model input: (B, 1, W, H), output: (B, 2, W, H)")

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

    def _init_preprocessing(self):
        """Initialize preprocessing components."""
        # # Gamma correction lookup table
        self.gamma_table = (255.0 * (np.linspace(0, 1, 256) ** self.GAMMA)).astype(
            np.uint8
        )

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=self.CLAHE_TILE_SIZE
        )

        if self.verbose:
            print(
                f"Preprocessing initialized: gamma={self.GAMMA}, "
                f"CLAHE(clip={self.CLAHE_CLIP_LIMIT}, tile={self.CLAHE_TILE_SIZE})"
            )

    def _extract_eye_region(self, frame, landmarks):
        """
        Extract left eye region from frame using MediaPipe landmarks.

        Args:
            frame: Input BGR frame
            landmarks: MediaPipe face landmarks

        Returns:
            tuple: (eye_crop, bbox) where bbox is (x, y, w, h), or (None, None)
        """
        h, w = frame.shape[:2]

        # Extract left eye landmark coordinates into pre-allocated buffer
        for i, idx in enumerate(self.LEFT_EYE_INDICES):
            landmark = landmarks.landmark[idx]
            self._eye_points_buffer[i, 0] = int(landmark.x * w)
            self._eye_points_buffer[i, 1] = int(landmark.y * h)

        if self.verbose:
            print(f"  First 3 eye landmarks: {self._eye_points_buffer[:3].tolist()}")

        # Compute bounding box using pre-allocated buffer
        x_min, y_min = self._eye_points_buffer.min(axis=0)
        x_max, y_max = self._eye_points_buffer.max(axis=0)

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

        return eye_crop, (x_min, y_min, bbox_w, bbox_h)

    def _preprocess(self, eye_crop):
        """
        Preprocess eye region for model inference.
        CRITICAL: Must match training preprocessing exactly.

        Uses pre-allocated buffers to avoid per-frame memory allocations.
        All operations use dst= parameters to write directly to pre-allocated buffers.

        Args:
            eye_crop: BGR image of eye region

        Returns:
            np.ndarray: Preprocessed tensor of shape (1, 1, 640, 400)
        """
        if self.verbose:
            t_start = time.time()

        # Step 1: Resize to model input size FIRST (into pre-allocated BGR buffer)
        # This ensures all subsequent operations work on fixed-size buffers
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

        # Step 2: Convert to grayscale (into pre-allocated buffer)
        cv2.cvtColor(self._eye_crop_resized, cv2.COLOR_BGR2GRAY, dst=self._gray_buffer)
        if self.verbose:
            t_gray = time.time()
            print(
                f"  Grayscale: {(t_gray - t_resize)*1000:.2f}ms, "
                f"shape={self._gray_buffer.shape}, range=[{self._gray_buffer.min()}, {self._gray_buffer.max()}]"
            )

        # # Step 3: Gamma correction (gamma=0.8) (into pre-allocated buffer)
        cv2.LUT(self._gray_buffer, self.gamma_table, dst=self._gamma_buffer)
        if self.verbose:
            t_gamma = time.time()
            print(
                f"  Gamma correction: {(t_gamma - t_gray)*1000:.2f}ms, "
                f"range=[{self._gamma_buffer.min()}, {self._gamma_buffer.max()}]"
            )

        # # Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization) (into pre-allocated buffer)
        self.clahe.apply(self._gamma_buffer, dst=self._resized_buffer)
        if self.verbose:
            t_clahe = time.time()
            print(
                f"  CLAHE: {(t_clahe - t_gamma)*1000:.2f}ms, "
                f"range=[{self._resized_buffer.min()}, {self._resized_buffer.max()}]"
            )

        # Step 5: Normalize (mean=0.5, std=0.5) -> range [-1, 1]
        # Use in-place operations with pre-allocated buffer
        np.multiply(self._resized_buffer, 1.0 / 255.0, out=self._normalized_buffer)
        np.subtract(
            self._normalized_buffer, self.NORMALIZE_MEAN, out=self._normalized_buffer
        )
        np.divide(
            self._normalized_buffer, self.NORMALIZE_STD, out=self._normalized_buffer
        )
        if self.verbose:
            t_normalize = time.time()
            print(
                f"  Normalize: {(t_normalize - t_clahe)*1000:.2f}ms, "
                f"range=[{self._normalized_buffer.min():.3f}, {self._normalized_buffer.max():.3f}]"
            )

        # Step 6: Fill pre-allocated tensor with transposed data
        # Model expects (B, C, W, H) = (1, 1, 640, 400), so transpose H,W -> W,H
        self._input_tensor[0, 0] = self._normalized_buffer.T

        if self.verbose:
            t_end = time.time()
            print(
                f"  Final tensor shape: {self._input_tensor.shape}, "
                f"total preprocessing: {(t_end - t_start)*1000:.2f}ms"
            )

        return self._input_tensor

    def _run_inference(self, input_tensor):
        """
        Run model inference on preprocessed input.

        Uses pre-allocated mask buffer to avoid per-frame allocations.

        Args:
            input_tensor: Preprocessed tensor of shape (1, 1, 640, 400)

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
        # Model outputs (B, C, W, H) = (1, 2, 640, 400), argmax over classes gives (640, 400)
        # Use pre-allocated _argmax_buffer to store result before transpose
        np.argmax(output_np[0], axis=0, out=self._argmax_buffer)
        # Transpose and copy to mask buffer (auto-casts int64/int32 to uint8)
        self._mask_buffer[:] = self._argmax_buffer.T

        inference_time = (time.time() - t_start) * 1000  # Convert to ms

        if self.verbose:
            print(
                f"  Inference: {inference_time:.1f}ms, "
                f"output shape={output_np.shape}, mask shape={self._mask_buffer.shape}, "
                f"mask values=[{self._mask_buffer.min()}, {self._mask_buffer.max()}]"
            )

        return self._mask_buffer, inference_time

    def _visualize(self, frame, eye_crop, mask, bbox, inference_time, face_detected):
        """
        Visualize segmentation results on frame.

        Uses pre-allocated buffers to minimize per-frame allocations.

        Args:
            frame: Original BGR frame
            eye_crop: Eye region crop
            mask: Binary segmentation mask (400, 640)
            bbox: Bounding box (x, y, w, h)
            inference_time: Inference time in milliseconds
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

        # Semi-transparent black background - draw directly on banner region only
        # Create a view of just the banner region for blending
        banner_region = annotated[
            banner_y : banner_y + banner_height, banner_x : banner_x + banner_w
        ]
        # Darken the banner region in-place (multiply by 0.5)
        np.multiply(banner_region, 0.5, out=banner_region, casting="unsafe")

        # Status text
        if not face_detected:
            status_text = "No Face Detected"
            status_color = (0, 255, 255)  # Yellow
        elif mask is None:
            status_text = "Move Closer"
            status_color = (0, 255, 255)  # Yellow
        else:
            status_text = "Face Detected"
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

        # If we have a valid mask, overlay it on the eye region
        if mask is not None and bbox is not None:
            x, y, w, h = bbox

            # Resize mask to match eye crop size (use view of pre-allocated buffer)
            mask_view = self._mask_viz_buffer[:h, :w]
            cv2.resize(mask, (w, h), dst=mask_view, interpolation=cv2.INTER_NEAREST)

            # Reuse green overlay cache if same size
            if self._green_overlay_size != (h, w):
                self._green_overlay_cache = np.zeros((h, w, 3), dtype=np.uint8)
                self._green_overlay_size = (h, w)
            else:
                self._green_overlay_cache.fill(0)

            self._green_overlay_cache[mask_view == 1] = (0, 255, 0)

            # Blend with original eye region
            eye_region = annotated[y : y + h, x : x + w]
            cv2.addWeighted(
                eye_region,
                1 - self.OVERLAY_ALPHA,
                self._green_overlay_cache,
                self.OVERLAY_ALPHA,
                0,
                eye_region,
            )

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

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
        provider_short = "GPU" if "CUDA" in self.execution_provider else "CPU"
        cv2.putText(
            annotated,
            f"Device: {provider_short}",
            (10, banner_height + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
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
        print("VisionAssist Live Demo")
        print("=" * 80)
        print(f"Model: {self.model_path}")
        print(f"Camera: {self.camera_index}")
        print(f"Execution Provider: {self.execution_provider}")
        print("\nControls:")
        print("  ESC - Exit")
        print("  SPACE - Pause/Resume")
        print("=" * 80 + "\n")

        # Create named window for proper display on macOS
        cv2.namedWindow("VisionAssist Live Demo", cv2.WINDOW_NORMAL)

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
            print("Check System Settings → Privacy & Security → Camera permissions.\n")

        try:
            while True:
                if not self.paused:
                    # Capture frame
                    if self.verbose:
                        t_capture = time.time()
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        # break
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

                    # Initialize variables
                    eye_crop = None
                    bbox = None
                    mask = None
                    inference_time = None

                    # Process if face detected
                    if face_detected:
                        landmarks = results.multi_face_landmarks[0]

                        # Extract eye region
                        eye_crop, bbox = self._extract_eye_region(frame, landmarks)

                        if eye_crop is not None:
                            # Preprocess
                            input_tensor = self._preprocess(eye_crop)

                            # Run inference
                            mask, inference_time = self._run_inference(input_tensor)

                    # Visualize
                    annotated = self._visualize(
                        frame, eye_crop, mask, bbox, inference_time, face_detected
                    )

                    # Display
                    cv2.imshow(
                        "TinyEfficientViT Semantic Segmentation with Preprocessing (Gamma + CLAHE)",
                        annotated,
                    )

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
    """Main entry point for the VisionAssist live demo."""
    parser = argparse.ArgumentParser(
        description="TinyEfficientViT Semantic Segmentation with Preprocessing (Gamma + CLAHE)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint file (REQUIRED)",
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
    demo = VisionAssistDemo(
        model_path=args.model, camera_index=args.camera, verbose=args.verbose
    )
    demo.run()


if __name__ == "__main__":
    main()
