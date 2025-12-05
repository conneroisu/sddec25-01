# Change: Add Precomputed Dataset HuggingFace Link

## Why

The training pipeline references a precomputed OpenEDS dataset hosted on HuggingFace (`Conner/openeds-precomputed`) that was generated using `training/precompute.py`. This dataset includes precomputed binary labels, spatial weights, and distance maps optimized for GPU training. The documentation currently describes the dataset acquisition and preprocessing workflow but doesn't provide a direct link to the precomputed version that's actually used in production training.

## What Changes

- **ADDED** reference to the precomputed HuggingFace dataset repository in the Dataset Acquisition subsection
- **ADDED** explanation that precomputed artifacts (spatial weights, distance maps) eliminate runtime preprocessing overhead
- **ADDED** citation/footnote with direct HuggingFace Hub link: `https://huggingface.co/datasets/Conner/openeds-precomputed`

## Impact

- Affected specs: `design-document` (existing capability, modification)
- Affected code: `sections/06-implementation.tex` (Training Dataset section, subsection 6.2.6)
- No code changes required
- Improves reproducibility by providing direct access to training data
