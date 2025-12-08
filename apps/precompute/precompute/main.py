"""
Dataset Precomputation Script for OpenEDS

This script downloads the OpenEDS dataset from Kaggle, applies all deterministic
preprocessing steps, and pushes the result to HuggingFace.

Preprocessing Pipeline:
1. Label Binarization: Convert 4-class to binary (0=background, 1=pupil)
2. Skip Empty Masks: Skip samples with no pupil in mask
3. Gamma Correction: Apply gamma=0.8 via LUT
4. CLAHE: Adaptive histogram equalization (clipLimit=1.5, tileGridSize=8x8)
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

from hf_dataset import create_hf_dataset_from_npz

# Constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 400
MAX_RADIUS = math.sqrt(IMAGE_WIDTH**2 + IMAGE_HEIGHT**2) / 2  # ~377.36
KAGGLE_DATASET_ID = "soumicksarker/openeds-dataset"

# Module-level CLAHE singleton for default parameters (clipLimit=1.5, tileGridSize=(8, 8))
_CLAHE = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

# Module-level gamma lookup table for default gamma=0.8
_GAMMA_08_TABLE = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)


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

    if num_classes == 2:
        # Optimized path for binary segmentation
        # Only need 2 transforms instead of 4 due to complementary relationship
        mask_1 = label == 1  # Pupil mask
        dist_inside_1 = distance_transform_edt(mask_1)
        dist_outside_1 = distance_transform_edt(~mask_1)

        # Class 1 (pupil): positive outside, negative inside
        dist_map[1] = dist_outside_1 - dist_inside_1
        # Class 0 (background): negation of class 1 (complementary)
        dist_map[0] = -dist_map[1]
    else:
        # General case for multi-class segmentation
        for c in range(num_classes):
            class_mask = label == c
            if class_mask.any():
                dist_inside = distance_transform_edt(class_mask)
                dist_outside = distance_transform_edt(~class_mask)
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

    # 6. Compute spatial weights
    spatial_weights = compute_spatial_weights(binary_label, sigma=5.0)

    # 7. Compute distance map
    dist_map = compute_dist_map(binary_label, num_classes=2)

    return {
        "image": image,
        "label": binary_label,
        "spatial_weights": spatial_weights,
        "dist_map": dist_map,
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


def has_existing_chunks(split_dir: Path) -> tuple[bool, int, int]:
    """
    Check if chunks already exist for a split and count samples.

    Args:
        split_dir: Path to the split directory (e.g., parquet_chunks/train)

    Returns:
        Tuple of (exists, chunk_count, sample_count)
    """
    chunks = sorted(split_dir.glob("chunk_*.npz"))
    if not chunks:
        return False, 0, 0

    total_samples = 0
    for chunk_path in chunks:
        with np.load(chunk_path) as data:
            total_samples += len(data['images'])

    return True, len(chunks), total_samples


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
            filenames=np.array(filenames),
        )

        # Clear accumulators in-place (don't reassign - that creates new local vars!)
        images.clear()
        labels.clear()
        spatial_weights.clear()
        dist_maps.clear()
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

    # Check for existing chunks
    train_dir = parquet_dir / "train"
    val_dir = parquet_dir / "validation"

    train_exists, train_chunks, train_count = has_existing_chunks(train_dir)
    val_exists, val_chunks, val_count = has_existing_chunks(val_dir)

    # Process train split
    if train_exists:
        print(f"Skipping train: {train_chunks} chunks with {train_count} samples already exist in {train_dir}", flush=True)
    else:
        train_count = process_and_save_split(train_pairs, "train", parquet_dir, chunk_size=500)

    # Process validation split
    if val_exists:
        print(f"Skipping validation: {val_chunks} chunks with {val_count} samples already exist in {val_dir}", flush=True)
    else:
        val_count = process_and_save_split(val_pairs, "validation", parquet_dir, chunk_size=500)

    print(f"\nParquet chunks saved to: {parquet_dir}", flush=True)
    print(f"  Train samples: {train_count}", flush=True)
    print(f"  Validation samples: {val_count}", flush=True)

    # Load npz files and create HuggingFace dataset (uses streaming generators)
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
