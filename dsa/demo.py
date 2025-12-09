#!/usr/bin/env python3
"""
VisionAssist Live Demo - DSA Semantic Segmentation

This demo application performs real-time semantic segmentation on webcam input
using the DeepSeek Sparse Attention (DSA) model. It demonstrates eye tracking
and pupil detection capabilities for the VisionAssist medical assistive project.

The application captures live video, runs inference using PyTorch with DSA,
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

from model import DSASegmentationModel, create_dsa_tiny, create_dsa_small, create_dsa_base


class VisionAssistDSADemo:
    """Main demo application for VisionAssist live webcam inference with DSA."""

    # MediaPipe left eye landmark indices (12 points around the eye)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466]

    # Target aspect ratio for eye region (width:height = 640:400 = 1.6:1)
    TARGET_ASPECT_RATIO = 640 / 400

    # Preprocessing parameters
    NORMALIZE_MEAN = 0.5
    NORMALIZE_STD = 0.5

    # Model input/output dimensions
    MODEL_WIDTH = 640
    MODEL_HEIGHT = 400

    # Display settings
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    MIN_EYE_REGION_SIZE = 100
    BBOX_PADDING = 0.2
    OVERLAY_ALPHA = 0.5

    def __init__(
        self,
        model_path: str,
        model_size: str = "small",
        camera_index: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize the VisionAssist DSA demo.

        Args:
            model_path: Path to PyTorch model checkpoint
            model_size: Model variant (tiny, small, base)
            camera_index: Camera device index
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

        # Pre-allocated buffers
        self._eye_crop_resized = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH, 3), dtype=np.uint8
        )
        self._gray_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        self._normalized_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.float32
        )
        self._input_tensor = np.empty(
            (1, 1, self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.float32
        )
        self._mask_buffer = np.empty(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8
        )
        self._eye_points_buffer = np.empty(
            (len(self.LEFT_EYE_INDICES), 2), dtype=np.int32
        )
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

        if platform.system() == "Darwin":
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.verbose:
            print(f"Camera initialized: {actual_width}x{actual_height}")

    def _init_model(self):
        """Initialize DSA model."""
        if self.verbose:
            print(f"Loading DSA model ({self.model_size}) from {self.model_path}...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.execution_provider = (
            "CUDAExecutionProvider"
            if self.device.type == "cuda"
            else "CPUExecutionProvider"
        )

        if self.verbose:
            print(f"Using device: {self.device}")

        # Create model architecture
        if self.model_size == "tiny":
            self.model = create_dsa_tiny(in_channels=1, num_classes=2)
        elif self.model_size == "small":
            self.model = create_dsa_small(in_channels=1, num_classes=2)
        elif self.model_size == "base":
            self.model = create_dsa_base(in_channels=1, num_classes=2)
        else:
            self.model = create_dsa_small(in_channels=1, num_classes=2)

        # Load checkpoint weights
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model loaded with {num_params:,} parameters")

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

    def _extract_eye_region(self, frame, landmarks):
        """Extract left eye region from frame using MediaPipe landmarks."""
        h, w = frame.shape[:2]

        for i, idx in enumerate(self.LEFT_EYE_INDICES):
            landmark = landmarks.landmark[idx]
            self._eye_points_buffer[i, 0] = int(landmark.x * w)
            self._eye_points_buffer[i, 1] = int(landmark.y * h)

        x_min, y_min = self._eye_points_buffer.min(axis=0)
        x_max, y_max = self._eye_points_buffer.max(axis=0)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        if bbox_w < self.MIN_EYE_REGION_SIZE or bbox_h < self.MIN_EYE_REGION_SIZE:
            return None, None

        pad_w = int(bbox_w * self.BBOX_PADDING)
        pad_h = int(bbox_h * self.BBOX_PADDING)

        x_min = max(0, x_min - pad_w)
        y_min = max(0, y_min - pad_h)
        x_max = min(w, x_max + pad_w)
        y_max = min(h, y_max + pad_h)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        current_ratio = bbox_w / bbox_h
        if current_ratio < self.TARGET_ASPECT_RATIO:
            target_w = int(bbox_h * self.TARGET_ASPECT_RATIO)
            diff = target_w - bbox_w
            x_min = max(0, x_min - diff // 2)
            x_max = min(w, x_max + diff // 2)
            bbox_w = x_max - x_min
        else:
            target_h = int(bbox_w / self.TARGET_ASPECT_RATIO)
            diff = target_h - bbox_h
            y_min = max(0, y_min - diff // 2)
            y_max = min(h, y_max + diff // 2)
            bbox_h = y_max - y_min

        eye_crop = frame[y_min:y_max, x_min:x_max]

        return eye_crop, (x_min, y_min, bbox_w, bbox_h)

    def _preprocess(self, eye_crop):
        """Preprocess eye region for model inference."""
        # Resize
        cv2.resize(
            eye_crop,
            (self.MODEL_WIDTH, self.MODEL_HEIGHT),
            dst=self._eye_crop_resized,
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert to grayscale
        cv2.cvtColor(self._eye_crop_resized, cv2.COLOR_BGR2GRAY, dst=self._gray_buffer)

        # Normalize
        np.multiply(self._gray_buffer, 1.0 / 255.0, out=self._normalized_buffer)
        np.subtract(
            self._normalized_buffer, self.NORMALIZE_MEAN, out=self._normalized_buffer
        )
        np.divide(
            self._normalized_buffer, self.NORMALIZE_STD, out=self._normalized_buffer
        )

        # Fill tensor (note: DSA model expects B, C, H, W not transposed)
        self._input_tensor[0, 0] = self._normalized_buffer

        return self._input_tensor

    def _run_inference(self, input_tensor):
        """Run DSA model inference."""
        t_start = time.time()

        input_torch = torch.from_numpy(input_tensor).to(self.device)

        with torch.no_grad():
            output = self.model(input_torch)

        output_np = output.cpu().numpy()

        # Argmax to get binary mask
        self._mask_buffer[:] = np.argmax(output_np[0], axis=0).astype(np.uint8)

        inference_time = (time.time() - t_start) * 1000

        return self._mask_buffer, inference_time

    def _visualize(self, frame, eye_crop, mask, bbox, inference_time, face_detected):
        """Visualize segmentation results on frame."""
        if self._overlay_buffer is None or self._overlay_buffer.shape != frame.shape:
            self._overlay_buffer = np.empty_like(frame)

        np.copyto(self._overlay_buffer, frame)
        annotated = self._overlay_buffer

        # Draw status banner
        banner_height = 50
        banner_region = annotated[:banner_height, :]
        np.multiply(banner_region, 0.5, out=banner_region, casting="unsafe")

        if not face_detected:
            status_text = "No Face Detected"
            status_color = (0, 255, 255)
        elif mask is None:
            status_text = "Move Closer"
            status_color = (0, 255, 255)
        else:
            status_text = "DSA Segmentation Active"
            status_color = (0, 255, 0)

        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (annotated.shape[1] - text_size[0]) // 2
        text_y = (banner_height + text_size[1]) // 2
        cv2.putText(
            annotated,
            status_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2,
        )

        # Overlay mask on eye region
        if mask is not None and bbox is not None:
            x, y, w, h = bbox

            mask_view = self._mask_viz_buffer[:h, :w]
            cv2.resize(mask, (w, h), dst=mask_view, interpolation=cv2.INTER_NEAREST)

            if self._green_overlay_size != (h, w):
                self._green_overlay_cache = np.zeros((h, w, 3), dtype=np.uint8)
                self._green_overlay_size = (h, w)
            else:
                self._green_overlay_cache.fill(0)

            self._green_overlay_cache[mask_view == 1] = (0, 255, 0)

            eye_region = annotated[y : y + h, x : x + w]
            cv2.addWeighted(
                eye_region,
                1 - self.OVERLAY_ALPHA,
                self._green_overlay_cache,
                self.OVERLAY_ALPHA,
                0,
                eye_region,
            )

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Draw stats
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

        cv2.putText(
            annotated,
            f"Model: DSA-{self.model_size}",
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
        print("VisionAssist Live Demo - DSA Segmentation")
        print("=" * 80)
        print(f"Model: DSA-{self.model_size} ({self.model_path})")
        print(f"Camera: {self.camera_index}")
        print(f"Execution Provider: {self.execution_provider}")
        print("\nControls:")
        print("  ESC - Exit")
        print("  SPACE - Pause/Resume")
        print("=" * 80 + "\n")

        cv2.namedWindow("VisionAssist DSA Demo", cv2.WINDOW_NORMAL)

        # Warm up camera
        print("Warming up camera...")
        for i in range(30):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.mean() > 1.0:
                print(f"Camera ready after {i+1} warmup frames.\n")
                break
            time.sleep(0.1)

        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue

                    # Convert to RGB for MediaPipe
                    if self._frame_rgb is None or self._frame_rgb.shape != frame.shape:
                        self._frame_rgb = np.empty_like(frame)
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb)

                    # Face detection
                    results = self.face_mesh.process(self._frame_rgb)
                    face_detected = results.multi_face_landmarks is not None

                    eye_crop = None
                    bbox = None
                    mask = None
                    inference_time = None

                    if face_detected:
                        landmarks = results.multi_face_landmarks[0]
                        eye_crop, bbox = self._extract_eye_region(frame, landmarks)

                        if eye_crop is not None:
                            input_tensor = self._preprocess(eye_crop)
                            mask, inference_time = self._run_inference(input_tensor)

                    annotated = self._visualize(
                        frame, eye_crop, mask, bbox, inference_time, face_detected
                    )

                    cv2.imshow("VisionAssist DSA Demo", annotated)
                    self.frame_count += 1
                else:
                    cv2.waitKey(100)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                elif key == ord(" "):
                    self.paused = not self.paused
                    print(f"\n{'Paused' if self.paused else 'Resumed'}")

        finally:
            self._cleanup()

    def _cleanup(self):
        """Release resources."""
        if hasattr(self, "cap"):
            self.cap.release()
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VisionAssist Live Demo - DSA Semantic Segmentation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint file (REQUIRED)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "base"],
        help="Model size variant",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable comprehensive logging",
    )

    args = parser.parse_args()

    demo = VisionAssistDSADemo(
        model_path=args.model,
        model_size=args.model_size,
        camera_index=args.camera,
        verbose=args.verbose,
    )
    demo.run()


if __name__ == "__main__":
    main()
