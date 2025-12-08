"""Memory-efficient HuggingFace dataset creation from NPZ chunks."""

import gc
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, Array2D, Array3D, load_dataset
from tqdm import tqdm


def create_hf_dataset_from_npz(
    output_dir: Path,
    image_height: int = 400,
    image_width: int = 640,
    verbose: bool = True,
) -> DatasetDict:
    """
    Create HuggingFace Dataset by converting npz chunks to parquet shards.

    This processes one chunk at a time to avoid OOM:
    1. Load a single npz chunk into memory
    2. Convert to Dataset and save as parquet shard
    3. Free memory before loading next chunk
    4. Load all parquet shards lazily (memory-mapped)

    Args:
        output_dir: Directory containing train/ and validation/ npz chunks
        image_height: Height of images (default: 400)
        image_width: Width of images (default: 640)
        verbose: Whether to print progress messages

    Returns:
        HuggingFace DatasetDict with train and validation splits
    """
    if verbose:
        print("Converting npz chunks to parquet shards...", flush=True)

    features = Features({
        "image": Array2D(shape=(image_height, image_width), dtype="uint8"),
        "label": Array2D(shape=(image_height, image_width), dtype="uint8"),
        "spatial_weights": Array2D(shape=(image_height, image_width), dtype="float32"),
        "dist_map": Array3D(shape=(2, image_height, image_width), dtype="float32"),
        "filename": Value("string"),
        "preprocessed": Value("bool"),
    })

    def convert_split_to_parquet(split_name: str) -> list[str]:
        """Convert npz chunks to parquet shards, return list of parquet file paths."""
        split_dir = output_dir / split_name
        parquet_dir = output_dir / f"{split_name}_parquet"

        # Check if parquet conversion already done
        if parquet_dir.exists():
            existing_parquets = sorted(parquet_dir.glob("*.parquet"))
            if existing_parquets:
                if verbose:
                    print(f"  {split_name}: Using {len(existing_parquets)} existing parquet shards", flush=True)
                return [str(p) for p in existing_parquets]

        parquet_dir.mkdir(parents=True, exist_ok=True)
        chunk_files = sorted(split_dir.glob("chunk_*.npz"))

        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {split_dir}")

        parquet_files = []
        iterator = tqdm(chunk_files, desc=f"  {split_name}") if verbose else chunk_files

        for chunk_file in iterator:
            # Load one chunk at a time
            with np.load(chunk_file) as data:
                n_samples = len(data["images"])

                # Build dict for this chunk only
                chunk_data = {
                    "image": list(data["images"]),
                    "label": list(data["labels"]),
                    "spatial_weights": list(data["spatial_weights"]),
                    "dist_map": list(data["dist_maps"]),
                    "filename": [str(f) for f in data["filenames"]],
                    "preprocessed": [True] * n_samples,
                }

            # Create small dataset from this single chunk
            chunk_ds = Dataset.from_dict(chunk_data, features=features)

            # Save as parquet shard
            parquet_path = parquet_dir / f"{chunk_file.stem}.parquet"
            chunk_ds.to_parquet(str(parquet_path))
            parquet_files.append(str(parquet_path))

            # Aggressive cleanup before next chunk
            del chunk_data, chunk_ds
            gc.collect()

        return parquet_files

    # Convert both splits to parquet
    train_parquets = convert_split_to_parquet("train")
    val_parquets = convert_split_to_parquet("validation")

    if verbose:
        print(f"Loading datasets from parquet shards...", flush=True)
        print(f"  train: {len(train_parquets)} shards", flush=True)
        print(f"  validation: {len(val_parquets)} shards", flush=True)

    # Load all parquet files - HuggingFace will memory-map them
    train_ds = load_dataset(
        "parquet",
        data_files=train_parquets,
        split="train",
    )

    val_ds = load_dataset(
        "parquet",
        data_files=val_parquets,
        split="train",  # load_dataset uses "train" as default split name
    )

    return DatasetDict({"train": train_ds, "validation": val_ds})
