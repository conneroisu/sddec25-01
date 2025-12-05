#!/usr/bin/env python3
"""
VisionAssist Live Demo - PyTorch TinyEfficientViT Semantic Segmentation

This demo application performs real-time semantic segmentation on webcam input
using the TinyEfficientViT model with native PyTorch inference. It supports
multiple device backends: CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU.

The application captures live video, runs inference using PyTorch,
and visualizes the segmentation results with overlays.

Usage:
    python demo_pytorch.py --model path/to/model.pt [--camera 0] [--device cuda|mps|cpu]

Key differences from demo.py (ONNX-based):
    - Uses native PyTorch inference instead of ONNX Runtime
    - Loads TinyEfficientViT model directly from .pt checkpoint
    - Supports MPS (Metal Performance Shaders) on Apple Silicon
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import platform
import torch

# Add parent directory to path for importing shared model
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.models.efficientvit import TinyEfficientViTSeg


def get_device(requested_device: str | None = None) -> torch.device:
    """
    Get the best available device for inference.

    Priority: CUDA > MPS > CPU (unless overridden by requested_device)

    Args:
        requested_device: Optional device name ("cuda", "mps", "cpu") to force

    Returns:
        torch.device for inference
    """
    if requested_device:
        device = torch.device(requested_device)
        # Validate requested device is available
        if requested_device == "cuda" and not torch.cuda.is_available():
            print(f"WARNING: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        if requested_device == "mps" and not torch.backends.mps.is_available():
            print(f"WARNING: MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class VisionAssistPyTorchDemo:
    """Main demo application for VisionAssist live webcam inference with PyTorch."""

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

    def __init__(
        self,
        model_path: str,
        camera_index: int = 0,
        verbose: bool = False,
        device: str | None = None,
    ):
        """
        Initialize the VisionAssist PyTorch demo.

        Args:
            model_path: Path to the PyTorch model file (.pt)
            camera_index: Camera device index (default 0)
            verbose: Enable comprehensive logging
            device: Force device ("cuda", "mps", "cpu"), default auto-detect
        """
        self.model_path = model_path
        self.camera_index = camera_index
        self.verbose = verbose

        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_frame_time = time.time()

        # Initialize device
        self.device = get_device(device)

        # Initialize components
        self._init_camera()
        self._init_model()
        self._init_face_mesh()
        self._init_preprocessing()

        # State
        self.paused = False
        self.frame_count = 0

        # Pre-allocated buffers for preprocessing (avoid per-frame allocations)
        self._eye_crop_resized = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH, 3), dtype=np.uint8
        )
        self._gray_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        self._gamma_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        self._resized_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        self._normalized_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.float32
        )
        self._input_tensor = np.empty(
            (1, 1, self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.float32
        )

        # Pre-allocated buffer for inference output
        self._mask_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
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
        """Initialize TinyEfficientViT model from PyTorch checkpoint."""
        if self.verbose:
            print(f"Loading TinyEfficientViT model from {self.model_path}...")

        # Create model with exact same configuration as training
        self.model = TinyEfficientViTSeg(
            in_channels=1,
            num_classes=2,
            embed_dims=(16, 32, 64),
            depths=(1, 1, 1),
            num_heads=(1, 1, 2),
            key_dims=(4, 4, 4),
            attn_ratios=(2, 2, 2),
            window_sizes=(7, 7, 7),
            mlp_ratios=(2, 2, 2),
            decoder_dim=32,
        )

        # Load weights from checkpoint
        state_dict = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())

        if self.verbose:
            print(f"Model loaded on device: {self.device.type.upper()}")
            print(f"Model parameters: {num_params:,}")

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
        # Gamma correction lookup table
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
            print(
                f"  First 3 eye landmarks: {self._eye_points_buffer[:3].tolist()}"
            )

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

        Pipeline (matches train_efficientvit.py):
        1. Resize to 640x400
        2. Convert to grayscale
        3. Gamma correction (gamma=0.8)
        4. CLAHE (clipLimit=1.5, tileGridSize=8x8)
        5. Normalize (mean=0.5, std=0.5)

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

        # Step 2: Convert to grayscale
        cv2.cvtColor(
            self._eye_crop_resized, cv2.COLOR_BGR2GRAY, dst=self._gray_buffer
        )

        # Step 3: Gamma correction (gamma=0.8)
        cv2.LUT(self._gray_buffer, self.gamma_table, dst=self._gamma_buffer)

        # Step 4: CLAHE
        self.clahe.apply(self._gamma_buffer, dst=self._resized_buffer)

        # Step 5: Normalize (mean=0.5, std=0.5) -> range [-1, 1]
        np.multiply(
            self._resized_buffer, 1.0 / 255.0, out=self._normalized_buffer
        )
        np.subtract(
            self._normalized_buffer, self.NORMALIZE_MEAN, out=self._normalized_buffer
        )
        np.divide(
            self._normalized_buffer, self.NORMALIZE_STD, out=self._normalized_buffer
        )

        # Step 6: Add batch and channel dimensions
        # Model expects (B, C, H, W) = (1, 1, 400, 640)
        self._input_tensor[0, 0] = self._normalized_buffer

        if self.verbose:
            t_end = time.time()
            print(
                f"  Preprocessing: {(t_end - t_start)*1000:.2f}ms, "
                f"shape={self._input_tensor.shape}"
            )

        return self._input_tensor

    def _run_inference(self, input_tensor):
        """
        Run model inference on preprocessed input.

        Args:
            input_tensor: Preprocessed tensor of shape (1, 1, 400, 640)

        Returns:
            tuple: (mask, inference_time) where mask is shape (400, 640)
        """
        t_start = time.time()

        # Convert numpy to torch tensor and move to device
        input_torch = torch.from_numpy(input_tensor).to(self.device)

        # Run inference with no gradient computation
        with torch.inference_mode():
            output = self.model(input_torch)

        # Post-processing: argmax to get binary mask
        # Model outputs (B, C, H, W) = (1, 2, 400, 640)
        output_np = output.cpu().numpy()
        mask = np.argmax(output_np[0], axis=0).astype(np.uint8)
        np.copyto(self._mask_buffer, mask)

        inference_time = (time.time() - t_start) * 1000

        if self.verbose:
            print(
                f"  Inference: {inference_time:.1f}ms, "
                f"output shape={output.shape}, "
                f"mask values=[{self._mask_buffer.min()}, {self._mask_buffer.max()}]"
            )

        return self._mask_buffer, inference_time

    def _visualize(
        self, frame, eye_crop, mask, bbox, inference_time, face_detected
    ):
        """
        Visualize segmentation results on frame.

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
        if (
            self._overlay_buffer is None
            or self._overlay_buffer.shape != frame.shape
        ):
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
        elif mask is None:
            status_text = "Move Closer"
            status_color = (0, 255, 255)  # Yellow
        else:
            status_text = "Face Detected"
            status_color = (0, 255, 0)  # Green

        text_size = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )[0]
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

            # Resize mask to match eye crop size
            mask_view = self._mask_viz_buffer[:h, :w]
            cv2.resize(
                mask, (w, h), dst=mask_view, interpolation=cv2.INTER_NEAREST
            )

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

        # Draw device (below inference time)
        cv2.putText(
            annotated,
            f"Device: {self.device.type.upper()}",
            (10, banner_height + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # Draw model type indicator (below device)
        cv2.putText(
            annotated,
            "Model: TinyEfficientViT",
            (10, banner_height + 155),
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
        print("VisionAssist Live Demo - PyTorch TinyEfficientViT")
        print("=" * 80)
        print(f"Model: {self.model_path}")
        print(f"Camera: {self.camera_index}")
        print(f"Device: {self.device.type.upper()}")
        print("\nControls:")
        print("  ESC - Exit")
        print("  SPACE - Pause/Resume")
        print("=" * 80 + "\n")

        # Create named window for proper display
        cv2.namedWindow("VisionAssist PyTorch Demo", cv2.WINDOW_NORMAL)

        # Allow camera to warm up
        print("Warming up camera...")
        for i in range(30):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.mean() > 1.0:
                    print(f"Camera ready after {i+1} warmup frames.\n")
                    break
            time.sleep(0.1)
        else:
            print("WARNING: Camera may not be capturing properly.")
            print(
                "Check System Settings -> Privacy & Security -> Camera permissions.\n"
            )

        try:
            while True:
                if not self.paused:
                    # Capture frame
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        continue

                    # Convert to RGB for MediaPipe
                    if (
                        self._frame_rgb is None
                        or self._frame_rgb.shape != frame.shape
                    ):
                        self._frame_rgb = np.empty_like(frame)
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb)

                    # Face detection
                    results = self.face_mesh.process(self._frame_rgb)
                    face_detected = results.multi_face_landmarks is not None

                    # Initialize variables
                    eye_crop = None
                    bbox = None
                    mask = None
                    inference_time = None

                    # Process if face detected
                    if face_detected:
                        landmarks = results.multi_face_landmarks[0]

                        # Extract eye region
                        eye_crop, bbox = self._extract_eye_region(
                            frame, landmarks
                        )

                        if eye_crop is not None:
                            # Preprocess
                            input_tensor = self._preprocess(eye_crop)

                            # Run inference
                            mask, inference_time = self._run_inference(
                                input_tensor
                            )

                    # Visualize
                    annotated = self._visualize(
                        frame,
                        eye_crop,
                        mask,
                        bbox,
                        inference_time,
                        face_detected,
                    )

                    # Display
                    cv2.imshow("VisionAssist PyTorch Demo", annotated)

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
    """Main entry point for the VisionAssist PyTorch live demo."""
    parser = argparse.ArgumentParser(
        description="VisionAssist Live Demo - PyTorch TinyEfficientViT Semantic Segmentation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint (.pt) (REQUIRED)",
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force device (default: auto-detect CUDA > MPS > CPU)",
    )

    args = parser.parse_args()

    # Initialize and run demo
    demo = VisionAssistPyTorchDemo(
        model_path=args.model,
        camera_index=args.camera,
        verbose=args.verbose,
        device=args.device,
    )
    demo.run()


if __name__ == "__main__":
    main()
