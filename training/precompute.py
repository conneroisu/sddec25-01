"""Precompute OpenEDS dataset and upload to HuggingFace Hub.
This script downloads the OpenEDS dataset from Kaggle, computes binary labels,
spatial weights, and distance maps, then packages and uploads to HuggingFace Hub.
Uses Parquet sharding to minimize memory usage (processes in chunks).
Usage:
    python precompute_dataset.py --hf-repo "username/openeds-precomputed" --hf-token "hf_xxx"
"""

import argparse
import gc
import os
import shutil
from typing import Generator
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from PIL import Image as PILImage
from scipy.ndimage import distance_transform_edt as distance
from tqdm import tqdm

CHUNK_SIZE = 500

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 640


def one_hot2dist(posmask: np.ndarray) -> np.ndarray:

    assert len(posmask.shape) == 2, f"Expected 2D mask, got shape {posmask.shape}"
    h, w = posmask.shape
    res = np.zeros_like(posmask, dtype=np.float32)
    posmask = posmask.astype(np.bool_)
    mxDist = np.sqrt((h - 1) ** 2 + (w - 1) ** 2)
    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res / mxDist


def compute_spatial_weights(label_binary: np.ndarray) -> np.ndarray:
    spatialWeights = cv2.Canny((label_binary * 255).astype(np.uint8), 0, 3) / 255
    spatialWeights = cv2.dilate(spatialWeights, (3, 3), iterations=1) * 20
    return spatialWeights.astype(np.float32)


def compute_distance_map(label_binary: np.ndarray) -> np.ndarray:

    distMap = np.stack(
        [one_hot2dist(label_binary == 0), one_hot2dist(label_binary == 1)], axis=0
    )
    return distMap.astype(np.float32)


def download_kaggle_dataset(output_path: str, username: str, key: str) -> str:

    if os.path.exists(output_path):
        print(f"Dataset already exists at: {output_path}")

        possible_paths = [
            output_path,
            os.path.join(output_path, "openEDS"),
            os.path.join(output_path, "openEDS", "openEDS"),
        ]
        for path in possible_paths:
            train_path = os.path.join(path, "train", "images")
            if os.path.exists(train_path):
                return path
        raise ValueError(f"Could not find train/images in {output_path}")

    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(kaggle_config, "w", encoding="utf-8") as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}')
    os.chmod(kaggle_config, 0o600)

    print("Downloading dataset from Kaggle...")
    result = os.system("kaggle datasets download -d soumicksarker/openeds-dataset")
    if result != 0:
        raise RuntimeError("Kaggle download failed. Check credentials.")

    print("Unzipping dataset...")
    os.system(f"unzip -q openeds-dataset.zip -d {output_path}")
    print(f"Dataset extracted to: {output_path}")

    possible_paths = [
        output_path,
        os.path.join(output_path, "openEDS"),
        os.path.join(output_path, "openEDS", "openEDS"),
    ]
    for path in possible_paths:
        train_path = os.path.join(path, "train", "images")
        if os.path.exists(train_path):
            return path
    raise ValueError(f"Could not find train/images in {output_path}")


def process_single_sample(
    images_path: str, labels_path: str, filename: str
) -> dict | None:

    basename = filename.replace(".png", "")

    img_path = os.path.join(images_path, filename)
    img = PILImage.open(img_path).convert("L")
    H, W = img.width, img.height

    label_path = os.path.join(labels_path, basename + ".npy")
    if not os.path.exists(label_path):
        return None
    label = np.load(label_path)
    label = np.resize(label, (W, H))

    label_binary = np.zeros_like(label, dtype=np.uint8)
    label_binary[label == 3] = 1

    spatial_weights = compute_spatial_weights(label_binary)

    dist_map = compute_distance_map(label_binary)
    return {
        "image": np.array(img),
        "label": label_binary,
        "spatial_weights": spatial_weights,
        "dist_map": dist_map,
        "filename": basename,
    }


def process_split_to_parquet(dataset_path: str, split: str, output_dir: str) -> str:

    split_path = os.path.join(dataset_path, split)
    images_path = os.path.join(split_path, "images")
    labels_path = os.path.join(split_path, "labels")
    if not os.path.exists(images_path):
        raise ValueError(f"Images path not found: {images_path}")
    if not os.path.exists(labels_path):
        raise ValueError(f"Labels path not found: {labels_path}")

    parquet_dir = os.path.join(output_dir, split)
    os.makedirs(parquet_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".png")])
    total_images = len(image_files)
    print(f"Found {total_images} images in {split} split")

    shard_idx = 0
    processed_count = 0
    skipped_shards = 0
    for chunk_start in range(0, total_images, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_images)
        chunk_files = image_files[chunk_start:chunk_end]

        parquet_path = os.path.join(parquet_dir, f"shard_{shard_idx:05d}.parquet")
        if os.path.exists(parquet_path):
            print(f"Shard {shard_idx} already exists, skipping...")
            skipped_shards += 1
            shard_idx += 1
            continue

        chunk_data = {
            "image": [],
            "label": [],
            "spatial_weights": [],
            "dist_map": [],
            "filename": [],
        }
        for filename in tqdm(chunk_files, desc=f"{split} shard {shard_idx}"):
            sample = process_single_sample(images_path, labels_path, filename)
            if sample is not None:

                chunk_data["image"].append(sample["image"].flatten())
                chunk_data["label"].append(sample["label"].flatten())
                chunk_data["spatial_weights"].append(
                    sample["spatial_weights"].flatten()
                )
                chunk_data["dist_map"].append(sample["dist_map"].flatten())
                chunk_data["filename"].append(sample["filename"])
                processed_count += 1

        if chunk_data["filename"]:
            table = pa.table(
                {
                    "image": chunk_data["image"],
                    "label": chunk_data["label"],
                    "spatial_weights": chunk_data["spatial_weights"],
                    "dist_map": chunk_data["dist_map"],
                    "filename": chunk_data["filename"],
                }
            )
            pq.write_table(table, parquet_path)
            print(f"Saved shard {shard_idx} with {len(chunk_data['filename'])} samples")

        del chunk_data
        gc.collect()
        shard_idx += 1
    print(
        f"Processed {processed_count} samples into {shard_idx - skipped_shards} new shards ({skipped_shards} skipped)"
    )
    return parquet_dir


def process_all_splits(dataset_path: str, output_dir: str) -> str:

    print("Processing training split...")
    train_parquet_dir = process_split_to_parquet(dataset_path, "train", output_dir)
    print(f"Training Parquet saved to: {train_parquet_dir}")
    print("\nProcessing validation split...")
    valid_parquet_dir = process_split_to_parquet(dataset_path, "validation", output_dir)
    print(f"Validation Parquet saved to: {valid_parquet_dir}")
    return output_dir


def upload_parquet_to_huggingface(parquet_dir: str, repo_id: str, token: str) -> None:

    print(f"Uploading parquet files to HuggingFace Hub: {repo_id}")
    api = HfApi()

    try:
        api.create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create repo: {e}")

    train_dir = os.path.join(parquet_dir, "train")
    if os.path.exists(train_dir):
        print("Uploading train split...")
        api.upload_folder(
            folder_path=train_dir,
            path_in_repo="data/train",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print("Train split uploaded.")

    valid_dir = os.path.join(parquet_dir, "validation")
    if os.path.exists(valid_dir):
        print("Uploading validation split...")
        api.upload_folder(
            folder_path=valid_dir,
            path_in_repo="data/validation",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print("Validation split uploaded.")
    print(
        f"Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Precompute OpenEDS dataset and upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/openeds-precomputed')",
    )
    parser.add_argument(
        "--hf-token", type=str, required=True, help="HuggingFace API token for upload"
    )
    parser.add_argument(
        "--kaggle-username",
        type=str,
        default="connerdohnesorge",
        help="Kaggle username for dataset download",
    )
    parser.add_argument(
        "--kaggle-key",
        type=str,
        default="e7a10abb1ed5e80b0b56394c05fb2f7c",
        help="Kaggle API key for dataset download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="openeds_data",
        help="Directory to store downloaded dataset",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only save parquet files locally, don't upload to HuggingFace",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip processing, just upload existing parquet files from --output-dir/parquet_shards",
    )
    args = parser.parse_args()
    parquet_output_dir = os.path.join(args.output_dir, "parquet_shards")
    if not args.upload_only:

        print("=" * 80)
        print("Step 1: Downloading OpenEDS dataset from Kaggle")
        print("=" * 80)
        dataset_path = download_kaggle_dataset(
            args.output_dir, args.kaggle_username, args.kaggle_key
        )
        print(f"Dataset path: {dataset_path}")

        print("\n" + "=" * 80)
        print("Step 2: Processing dataset splits (Parquet shard, memory efficient)")
        print("=" * 80)
        process_all_splits(dataset_path, parquet_output_dir)
        print(f"\nParquet files saved to: {parquet_output_dir}")
    else:
        print("=" * 80)
        print("Skipping download and processing (--upload-only)")
        print("=" * 80)
        print(f"Using existing parquet files from: {parquet_output_dir}")

    if not args.local_only:
        print("\n" + "=" * 80)
        print("Step 3: Uploading to HuggingFace Hub")
        print("=" * 80)
        upload_parquet_to_huggingface(parquet_output_dir, args.hf_repo, args.hf_token)
    print("\n" + "=" * 80)
    print("Precomputation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
