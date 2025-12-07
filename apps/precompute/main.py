"""
Dataset Precomputation Script for OpenEDS

This script downloads the OpenEDS dataset from Kaggle, applies all deterministic
preprocessing steps, and pushes the result to HuggingFace.

Preprocessing Pipeline:
1. Label Binarization: Convert 4-class to binary (0=background, 1=pupil)
2. Skip Empty Masks: Skip samples with no pupil in mask
3. Gamma Correction: Apply gamma=0.8 via LUT
4. CLAHE: Adaptive histogram equalization (clipLimit=1.5, tileGridSize=8x8)
5. Ellipse Parameter Extraction: Fit ellipse to pupil contour
6. Spatial Weights: Compute boundary weights using morphological gradient
7. Distance Map: Signed distance transform per class

Usage:
    # Full precompute and push to HuggingFace
    python precompute.py --hf-repo Conner/sddec25-01

    # Local only (no HuggingFace push)
    python precompute.py --no-push --output-dir ./my_dataset

    # Validate existing dataset
    python precompute.py --validate --hf-repo Conner/sddec25-01

    # Skip Kaggle download (use existing files)
    python precompute.py --skip-download --output-dir ./existing_data
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# Constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 400
MAX_RADIUS = math.sqrt(IMAGE_WIDTH**2 + IMAGE_HEIGHT**2) / 2  # ~377.36
KAGGLE_DATASET_ID = "soumicksarker/openeds-dataset"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute OpenEDS dataset and push to HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--hf-repo",
        type=str,
        default="Conner/sddec25-01",
        help="HuggingFace repository to push to",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./precompute_output",
        help="Local output directory for intermediate files",
    )

    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to HuggingFace",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing dataset preprocessing",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download (use existing files)",
    )

    return parser.parse_args()


# =============================================================================
# Preprocessing Functions
# =============================================================================


def binarize_label(raw_label: np.ndarray) -> np.ndarray:
    """
    Convert 4-class label to binary label.

    Input classes:
        0 = background
        1 = sclera
        2 = iris
        3 = pupil

    Output classes:
        0 = everything else (background + sclera + iris)
        1 = pupil only

    Args:
        raw_label: Raw 4-class segmentation mask [H, W]

    Returns:
        Binary mask [H, W] with 0=background, 1=pupil
    """
    return (raw_label == 3).astype(np.uint8)


def is_empty_mask(binary_label: np.ndarray) -> bool:
    """
    Check if binary mask is empty (no pupil pixels).

    Args:
        binary_label: Binary segmentation mask [H, W]

    Returns:
        True if mask is empty, False otherwise
    """
    return np.sum(binary_label) == 0


def apply_gamma_correction(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """
    Apply gamma correction using a lookup table.

    Args:
        image: Grayscale image [H, W] uint8
        gamma: Gamma value (default 0.8)

    Returns:
        Gamma-corrected image [H, W] uint8
    """
    gamma_table = 255.0 * (np.linspace(0, 1, 256) ** gamma)
    gamma_table = gamma_table.astype(np.uint8)
    return cv2.LUT(image, gamma_table)


def apply_clahe(
    image: np.ndarray, clip_limit: float = 1.5, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image: Grayscale image [H, W] uint8
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        CLAHE-enhanced image [H, W] uint8
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def extract_ellipse_params(
    binary_mask: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Extract ellipse parameters from binary mask.

    Fits an ellipse to the largest contour in the mask. Falls back to
    moments-based circle fitting if ellipse fitting fails or contour
    is degenerate (collinear points can cause cv2.fitEllipse to crash).

    Args:
        binary_mask: Binary segmentation mask [H, W]

    Returns:
        Tuple of (cx, cy, rx, ry) - center and radii in pixels
        Returns (0, 0, 0, 0) if no valid contour found
    """
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    def moments_fallback(contour) -> tuple[float, float, float, float]:
        """Fallback to moments-based circle fitting."""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            area = cv2.contourArea(contour)
            radius = math.sqrt(area / math.pi) if area > 0 else 0.0
            return cx, cy, radius, radius
        return 0.0, 0.0, 0.0, 0.0

    # Need at least 5 points for ellipse fitting
    if len(largest_contour) < 5:
        return moments_fallback(largest_contour)

    # Check for degenerate contours that can crash cv2.fitEllipse
    contour_points = largest_contour.reshape(-1, 2).astype(np.float64)

    # Check spread in both dimensions - thin contours can cause issues
    x_spread = contour_points[:, 0].max() - contour_points[:, 0].min()
    y_spread = contour_points[:, 1].max() - contour_points[:, 1].min()

    if x_spread < 2.0 or y_spread < 2.0:
        return moments_fallback(largest_contour)

    # Check if contour has reasonable aspect ratio (not too elongated)
    if max(x_spread, y_spread) / max(min(x_spread, y_spread), 1.0) > 50.0:
        return moments_fallback(largest_contour)

    # Check if area is too small (could indicate degenerate contour)
    area = cv2.contourArea(largest_contour)
    if area < 10.0:
        return moments_fallback(largest_contour)

    # Check for collinearity using covariance eigenvalues
    # Collinear points cause cv2.fitEllipse to segfault at C++ level
    if len(contour_points) >= 3:
        mean = contour_points.mean(axis=0)
        centered = contour_points - mean
        cov = np.cov(centered.T)
        eigenvals = np.linalg.eigvalsh(cov)
        # If smallest eigenvalue is near zero, points are nearly collinear
        # Use conservative threshold of 5.0 to catch borderline cases
        if eigenvals.min() < 5.0:
            return moments_fallback(largest_contour)
        # Check condition number - high values indicate near-singularity
        if eigenvals.max() / max(eigenvals.min(), 1e-10) > 100.0:
            return moments_fallback(largest_contour)

    try:
        ellipse = cv2.fitEllipse(largest_contour)
        (cx, cy), (width, height), angle = ellipse

        # Validate output - check for NaN/Inf and positive dimensions
        if not np.isfinite([cx, cy, width, height]).all():
            return moments_fallback(largest_contour)
        if width <= 0 or height <= 0:
            return moments_fallback(largest_contour)

        rx = width / 2.0
        ry = height / 2.0
        return cx, cy, rx, ry
    except cv2.error:
        return moments_fallback(largest_contour)


def normalize_ellipse_params(
    cx: float, cy: float, rx: float, ry: float
) -> tuple[float, float, float, float]:
    """
    Normalize ellipse parameters to [0, 1] range.

    Args:
        cx: Center x in pixels
        cy: Center y in pixels
        rx: X-radius in pixels
        ry: Y-radius in pixels

    Returns:
        Normalized (cx_norm, cy_norm, rx_norm, ry_norm)
    """
    cx_norm = cx / IMAGE_WIDTH
    cy_norm = cy / IMAGE_HEIGHT
    rx_norm = rx / MAX_RADIUS
    ry_norm = ry / MAX_RADIUS
    return cx_norm, cy_norm, rx_norm, ry_norm


def compute_spatial_weights(label: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """
    Compute spatial weights using morphological boundary detection.

    Creates boundary weights by computing the morphological gradient
    (dilation - erosion) and smoothing with a Gaussian filter.

    Args:
        label: Binary segmentation mask [H, W]
        sigma: Gaussian smoothing sigma

    Returns:
        Spatial weights [H, W] float32
    """
    dilated = ndimage.binary_dilation(label, iterations=1)
    eroded = ndimage.binary_erosion(label, iterations=1)
    boundary = dilated.astype(float) - eroded.astype(float)
    weights = ndimage.gaussian_filter(boundary, sigma=sigma)
    return weights.astype(np.float32)


def compute_dist_map(label: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Compute signed distance transform per class.

    For each class, computes the signed distance where:
    - Positive values: outside the class region
    - Negative values: inside the class region

    The distance is normalized by the diagonal of the image.

    Args:
        label: Binary segmentation mask [H, W]
        num_classes: Number of classes (default 2 for binary)

    Returns:
        Distance map [num_classes, H, W] float32, normalized to [-1, 1]
    """
    dist_map = np.zeros((num_classes, *label.shape), dtype=np.float32)

    for c in range(num_classes):
        class_mask = label == c
        if class_mask.any():
            dist_inside = distance_transform_edt(class_mask)
            dist_outside = distance_transform_edt(~class_mask)
            # Signed distance: positive outside, negative inside
            dist_map[c] = dist_outside - dist_inside
        else:
            # If class is not present, distance is everywhere
            dist_map[c] = distance_transform_edt(np.ones_like(label))

    # Normalize by image diagonal
    max_dist = np.sqrt(label.shape[0] ** 2 + label.shape[1] ** 2)
    dist_map = dist_map / max_dist

    return dist_map


def preprocess_sample(
    image: np.ndarray, raw_label: np.ndarray, filename: str
) -> Optional[dict[str, Any]]:
    """
    Apply full preprocessing pipeline to a single sample.

    Pipeline:
    1. Binarize label (4-class -> binary)
    2. Check for empty mask (skip if empty)
    3. Gamma correction
    4. CLAHE
    5. Extract ellipse parameters
    6. Compute spatial weights
    7. Compute distance map

    Args:
        image: Raw grayscale image [H, W] uint8
        raw_label: Raw 4-class segmentation mask [H, W]
        filename: Original filename for traceability

    Returns:
        Dictionary with all preprocessed fields, or None if sample should be skipped
    """
    # 1. Binarize label
    binary_label = binarize_label(raw_label)

    # 2. Skip if empty mask
    if is_empty_mask(binary_label):
        return None

    # 3. Gamma correction
    gamma_image = apply_gamma_correction(image, gamma=0.8)

    # 4. CLAHE
    clahe_image = apply_clahe(gamma_image, clip_limit=1.5, tile_grid_size=(8, 8))

    # 5. Extract and normalize ellipse parameters
    cx, cy, rx, ry = extract_ellipse_params(binary_label)
    cx_norm, cy_norm, rx_norm, ry_norm = normalize_ellipse_params(cx, cy, rx, ry)

    # 6. Compute spatial weights
    spatial_weights = compute_spatial_weights(binary_label, sigma=5.0)

    # 7. Compute distance map
    dist_map = compute_dist_map(binary_label, num_classes=2)

    return {
        "image": clahe_image,
        "label": binary_label,
        "spatial_weights": spatial_weights,
        "dist_map": dist_map,
        "cx": float(cx_norm),
        "cy": float(cy_norm),
        "rx": float(rx_norm),
        "ry": float(ry_norm),
        "filename": filename,
        "preprocessed": True,
    }


# =============================================================================
# Dataset Loading and Processing
# =============================================================================


def download_kaggle_dataset() -> str:
    """
    Download OpenEDS dataset from Kaggle using kagglehub.

    Returns:
        Path to downloaded dataset directory
    """
    import kagglehub

    print(f"Downloading dataset from Kaggle: {KAGGLE_DATASET_ID}", flush=True)
    path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    print(f"Dataset downloaded to: {path}", flush=True)
    return path


def find_dataset_files(
    dataset_path: str,
    val_ratio: float = 0.1,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Find all image/label pairs in the dataset directory.

    The OpenEDS Kaggle dataset is organized by subject folders (S_*) with
    .png images and .npy labels. Only a subset of images have labels
    (semantic segmentation subset).

    Expected structure:
        dataset_path/
            openEDS/
                openEDS/
                    images.txt      # List of all images
                    labels.txt      # List of labeled images (subset)
                    S_0/
                        0.png       # Image
                        0.npy       # Label (4-class segmentation)
                        1.png
                        ...
                    S_1/
                    ...

    Args:
        dataset_path: Path to the downloaded dataset
        val_ratio: Fraction of subjects to use for validation (default 0.1)

    Returns:
        Tuple of (train_pairs, val_pairs) where each pair is (image_path, label_path)
    """
    base_path = Path(dataset_path)

    # Find the openEDS/openEDS directory
    possible_roots = [
        base_path / "openEDS" / "openEDS",
        base_path / "openEDS",
        base_path,
    ]

    root = None
    for possible_root in possible_roots:
        # Check for subject folders (S_*)
        subject_folders = list(possible_root.glob("S_*"))
        if subject_folders:
            root = possible_root
            break

    if root is None:
        raise FileNotFoundError(
            f"Could not find dataset structure in {dataset_path}. "
            f"Expected openEDS/openEDS/ with S_* subject folders."
        )

    print(f"Found dataset root at: {root}", flush=True)

    # Find all image/label pairs from subject folders
    all_pairs = []
    subject_to_pairs: dict[str, list[tuple[str, str]]] = {}

    # Get all subject folders
    subject_folders = sorted(root.glob("S_*"))
    print(f"Found {len(subject_folders)} subject folders", flush=True)

    for subject_folder in subject_folders:
        subject_name = subject_folder.name
        subject_to_pairs[subject_name] = []

        # Find all .npy label files in this subject folder
        label_files = sorted(subject_folder.glob("*.npy"))

        for label_file in label_files:
            # Find corresponding image file
            image_name = label_file.stem + ".png"
            image_file = subject_folder / image_name

            if image_file.exists():
                pair = (str(image_file), str(label_file))
                all_pairs.append(pair)
                subject_to_pairs[subject_name].append(pair)

    print(f"Found {len(all_pairs)} total image/label pairs", flush=True)

    # Split by subjects (not by individual images) for proper validation
    subjects = sorted(subject_to_pairs.keys())
    num_val_subjects = max(1, int(len(subjects) * val_ratio))

    # Use last N subjects for validation (deterministic split)
    val_subjects = set(subjects[-num_val_subjects:])
    train_subjects = set(subjects[:-num_val_subjects])

    train_pairs = []
    val_pairs = []

    for subject, pairs in subject_to_pairs.items():
        if subject in val_subjects:
            val_pairs.extend(pairs)
        else:
            train_pairs.extend(pairs)

    print(f"Split: {len(train_subjects)} train subjects, {len(val_subjects)} val subjects", flush=True)
    print(f"Found {len(train_pairs)} train pairs, {len(val_pairs)} validation pairs", flush=True)

    return train_pairs, val_pairs


def load_image(image_path: str) -> np.ndarray:
    """Load grayscale image from path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def load_label(label_path: str) -> np.ndarray:
    """Load label from path (.npy or .png)."""
    if label_path.endswith(".npy"):
        return np.load(label_path).astype(np.uint8)
    else:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Could not load label: {label_path}")
        return label.astype(np.uint8)


def process_split_generator(
    pairs: list[tuple[str, str]], split_name: str
):
    """
    Generator that yields preprocessed samples one at a time.

    Uses generator pattern to avoid loading all samples into memory.

    Args:
        pairs: List of (image_path, label_path) tuples
        split_name: Name of the split (train/validation)

    Yields:
        Preprocessed sample dictionaries
    """
    skipped = 0
    processed = 0

    for image_path, label_path in tqdm(pairs, desc=f"Processing {split_name}"):
        try:
            image = load_image(image_path)
            raw_label = load_label(label_path)

            # Get filename from path
            filename = Path(image_path).name

            # Preprocess
            sample = preprocess_sample(image, raw_label, filename)

            if sample is None:
                skipped += 1
                continue

            processed += 1
            yield sample

        except Exception as e:
            print(f"Error processing {image_path}: {e}", flush=True)
            skipped += 1

    print(f"{split_name}: Processed {processed} samples, skipped {skipped}", flush=True)


# =============================================================================
# HuggingFace Upload
# =============================================================================


def process_and_save_split(
    pairs: list[tuple[str, str]],
    split_name: str,
    output_dir: Path,
    chunk_size: int = 100,
) -> int:
    """
    Process samples and save directly to npz files in chunks.

    This avoids loading all samples into memory by writing to disk periodically.
    Uses numpy's compressed npz format for efficient storage.

    Args:
        pairs: List of (image_path, label_path) tuples
        split_name: Name of the split (train/validation)
        output_dir: Directory to save chunk files
        chunk_size: Number of samples per chunk file

    Returns:
        Total number of processed samples
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators for current chunk - use lists of numpy arrays
    images = []
    labels = []
    spatial_weights = []
    dist_maps = []
    cxs = []
    cys = []
    rxs = []
    rys = []
    filenames = []

    chunk_idx = 0
    total_processed = 0
    skipped = 0

    def save_chunk():
        nonlocal chunk_idx

        if not images:
            return

        # Stack arrays and save as compressed npz
        np.savez_compressed(
            split_dir / f"chunk_{chunk_idx:05d}.npz",
            images=np.stack(images),
            labels=np.stack(labels),
            spatial_weights=np.stack(spatial_weights),
            dist_maps=np.stack(dist_maps),
            cx=np.array(cxs, dtype=np.float32),
            cy=np.array(cys, dtype=np.float32),
            rx=np.array(rxs, dtype=np.float32),
            ry=np.array(rys, dtype=np.float32),
            filenames=np.array(filenames),
        )

        # Clear accumulators in-place (don't reassign - that creates new local vars!)
        images.clear()
        labels.clear()
        spatial_weights.clear()
        dist_maps.clear()
        cxs.clear()
        cys.clear()
        rxs.clear()
        rys.clear()
        filenames.clear()
        chunk_idx += 1

    for image_path, label_path in tqdm(pairs, desc=f"Processing {split_name}"):
        try:
            image = load_image(image_path)
            raw_label = load_label(label_path)
            filename = Path(image_path).name

            sample = preprocess_sample(image, raw_label, filename)

            if sample is None:
                skipped += 1
                continue

            # Append to accumulators (keep as numpy arrays)
            images.append(sample["image"])
            labels.append(sample["label"])
            spatial_weights.append(sample["spatial_weights"])
            dist_maps.append(sample["dist_map"])
            cxs.append(sample["cx"])
            cys.append(sample["cy"])
            rxs.append(sample["rx"])
            rys.append(sample["ry"])
            filenames.append(sample["filename"])
            total_processed += 1

            # Write chunk to disk when full
            if len(images) >= chunk_size:
                save_chunk()

        except Exception as e:
            print(f"Error processing {image_path}: {e}", flush=True)
            skipped += 1

    # Write remaining samples
    save_chunk()

    print(f"{split_name}: Processed {total_processed} samples, skipped {skipped}", flush=True)
    return total_processed


def create_hf_dataset_from_npz(output_dir: Path):
    """
    Create HuggingFace Dataset from saved npz chunk files.

    Loads the chunked npz files and converts to HuggingFace Dataset format.
    WARNING: This loads everything into memory. For large datasets, upload
    npz chunks directly instead.

    Args:
        output_dir: Directory containing train/ and validation/ npz chunks

    Returns:
        HuggingFace DatasetDict with train and validation splits
    """
    from datasets import Dataset, DatasetDict, Features, Value, Array2D, Array3D

    print("Loading datasets from npz files...", flush=True)

    def load_split(split_name: str) -> Dataset:
        split_dir = output_dir / split_name
        chunk_files = sorted(split_dir.glob("chunk_*.npz"))

        all_samples = []
        for chunk_file in tqdm(chunk_files, desc=f"Loading {split_name}"):
            data = np.load(chunk_file)
            n_samples = len(data["cx"])

            for i in range(n_samples):
                all_samples.append({
                    "image": data["images"][i],
                    "label": data["labels"][i],
                    "spatial_weights": data["spatial_weights"][i],
                    "dist_map": data["dist_maps"][i],
                    "cx": float(data["cx"][i]),
                    "cy": float(data["cy"][i]),
                    "rx": float(data["rx"][i]),
                    "ry": float(data["ry"][i]),
                    "filename": str(data["filenames"][i]),
                    "preprocessed": True,
                })

        # Define features schema
        features = Features({
            "image": Array2D(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype="uint8"),
            "label": Array2D(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype="uint8"),
            "spatial_weights": Array2D(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype="float32"),
            "dist_map": Array3D(shape=(2, IMAGE_HEIGHT, IMAGE_WIDTH), dtype="float32"),
            "cx": Value("float32"),
            "cy": Value("float32"),
            "rx": Value("float32"),
            "ry": Value("float32"),
            "filename": Value("string"),
            "preprocessed": Value("bool"),
        })

        return Dataset.from_list(all_samples, features=features)

    train_dataset = load_split("train")
    val_dataset = load_split("validation")

    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def push_to_huggingface(dataset, repo_id: str):
    """
    Push dataset to HuggingFace Hub.

    Args:
        dataset: HuggingFace DatasetDict
        repo_id: Repository ID (e.g., "Conner/sddec25-01")
    """
    from huggingface_hub import HfApi

    print(f"Pushing dataset to HuggingFace: {repo_id}", flush=True)

    # Check authentication
    api = HfApi()
    try:
        api.whoami()
    except Exception:
        print(
            "Error: Not logged in to HuggingFace. "
            "Run `huggingface-cli login` first.",
            flush=True,
        )
        sys.exit(1)

    # Push with chunked upload for large datasets
    dataset.push_to_hub(
        repo_id,
        private=False,
        commit_message="Update preprocessed OpenEDS dataset",
    )

    print(f"Successfully pushed to https://huggingface.co/datasets/{repo_id}", flush=True)


# =============================================================================
# Validation Mode
# =============================================================================


def validate_dataset(repo_id: str, num_samples: int = 10):
    """
    Validate that preprocessed dataset matches runtime processing.

    Compares preprocessed images with images processed at runtime to
    ensure they match pixel-level.

    Args:
        repo_id: HuggingFace repository ID
        num_samples: Number of samples to validate
    """
    from datasets import load_dataset

    print(f"Validating dataset: {repo_id}", flush=True)

    # Load preprocessed dataset
    dataset = load_dataset(repo_id, split="train")

    # Check for preprocessed flag
    if "preprocessed" not in dataset.column_names:
        print("Error: Dataset does not have 'preprocessed' column", flush=True)
        return False

    # Download raw dataset for comparison
    raw_path = download_kaggle_dataset()
    train_pairs, _ = find_dataset_files(raw_path)

    # Build filename to path mapping
    filename_to_path = {}
    for image_path, label_path in train_pairs:
        filename = Path(image_path).name
        filename_to_path[filename] = (image_path, label_path)

    # Validate samples
    validated = 0
    failed = 0

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        filename = sample["filename"]

        if filename not in filename_to_path:
            print(f"Warning: Could not find raw file for {filename}", flush=True)
            continue

        image_path, label_path = filename_to_path[filename]

        # Load and process raw data
        raw_image = load_image(image_path)
        raw_label = load_label(label_path)

        processed = preprocess_sample(raw_image, raw_label, filename)

        if processed is None:
            print(f"Warning: Sample {filename} would be skipped by preprocessing", flush=True)
            continue

        # Compare preprocessed vs runtime processed
        stored_image = np.array(sample["image"])
        runtime_image = processed["image"]

        if np.array_equal(stored_image, runtime_image):
            validated += 1
        else:
            diff = np.abs(stored_image.astype(int) - runtime_image.astype(int))
            max_diff = np.max(diff)
            print(f"Mismatch in {filename}: max pixel diff = {max_diff}", flush=True)
            failed += 1

    print(f"Validation complete: {validated} passed, {failed} failed", flush=True)
    return failed == 0


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point."""
    args = parse_args()

    # Validation mode
    if args.validate:
        success = validate_dataset(args.hf_repo)
        sys.exit(0 if success else 1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download or use existing dataset
    if args.skip_download:
        dataset_path = str(output_dir)
        print(f"Using existing files in: {dataset_path}", flush=True)
    else:
        dataset_path = download_kaggle_dataset()

    # Find all image/label pairs
    train_pairs, val_pairs = find_dataset_files(dataset_path)

    if not train_pairs and not val_pairs:
        print("Error: No valid image/label pairs found", flush=True)
        sys.exit(1)

    # Process and save to parquet chunks (memory efficient)
    print(f"\nProcessing {len(train_pairs)} train and {len(val_pairs)} validation pairs...", flush=True)
    parquet_dir = output_dir / "parquet_chunks"

    train_count = process_and_save_split(train_pairs, "train", parquet_dir, chunk_size=500)
    val_count = process_and_save_split(val_pairs, "validation", parquet_dir, chunk_size=500)

    print(f"\nParquet chunks saved to: {parquet_dir}", flush=True)
    print(f"  Train samples: {train_count}", flush=True)
    print(f"  Validation samples: {val_count}", flush=True)

    # Load npz files and create HuggingFace dataset
    # NOTE: This may OOM for very large datasets
    hf_dataset = create_hf_dataset_from_npz(parquet_dir)

    # Print dataset info
    print("\nDataset Summary:", flush=True)
    print(f"  Train samples: {len(hf_dataset['train'])}", flush=True)
    print(f"  Validation samples: {len(hf_dataset['validation'])}", flush=True)
    print(f"  Features: {list(hf_dataset['train'].features.keys())}", flush=True)

    # Save locally
    local_path = output_dir / "dataset"
    print(f"\nSaving dataset locally to: {local_path}", flush=True)
    hf_dataset.save_to_disk(str(local_path))

    # Push to HuggingFace
    if not args.no_push:
        push_to_huggingface(hf_dataset, args.hf_repo)
    else:
        print("Skipping HuggingFace push (--no-push flag set)", flush=True)

    print("\nPrecomputation complete!", flush=True)


if __name__ == "__main__":
    main()
