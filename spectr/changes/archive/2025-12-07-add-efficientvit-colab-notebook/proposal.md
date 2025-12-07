# Change: Add Google Colab Notebook for EfficientViT Segmentation Training

## Why
The current `train_efficientvit.py` script requires Modal infrastructure for cloud execution, which has cost and setup barriers. A Google Colab notebook would provide:
- Free GPU access via Google Colab
- Interactive experimentation and visualization
- Lower barrier to entry for team members and external contributors
- No need for Modal secrets or Databricks MLflow configuration

## What Changes
- Add `training/train_efficientvit.ipynb` - a Jupyter notebook version of the TinyEfficientViT segmentation training script
- Remove Modal-specific code (image definitions, volume mounts, secrets)
- Replace MLflow/Databricks logging with Colab-friendly alternatives (TensorBoard or simple matplotlib plots)
- Add Colab-specific setup cells (GPU check, pip installs, HuggingFace dataset download)
- Preserve the core training logic: TinyEfficientViTSeg model architecture, CombinedLoss, augmentations, metrics

## Impact
- Affected specs: `training`
- Affected code: New file `training/train_efficientvit.ipynb`
- No breaking changes to existing `train_efficientvit.py` (Modal version remains unchanged)
