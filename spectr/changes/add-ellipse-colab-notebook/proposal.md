# Change: Add Google Colab Notebook for Ellipse Regression Training

## Why
The current `train_ellipse.py` script requires Modal infrastructure for cloud execution, which has cost and setup barriers. A Google Colab notebook would provide:
- Free GPU access via Google Colab
- Interactive experimentation and visualization
- Lower barrier to entry for team members and external contributors
- No need for Modal secrets or Databricks MLflow configuration

## What Changes
- Add `training/train_ellipse.ipynb` - a Jupyter notebook version of the ellipse regression training script
- Remove Modal-specific code (image definitions, volume mounts, secrets)
- Replace MLflow/Databricks logging with Colab-friendly alternatives (TensorBoard or simple file logging)
- Add Colab-specific setup cells (GPU check, pip installs, HuggingFace dataset download)
- Preserve the core training logic: model architecture, loss functions, augmentations, metrics

## Impact
- Affected specs: `training`
- Affected code: New file `training/train_ellipse.ipynb`
- No breaking changes to existing `train_ellipse.py` (Modal version remains unchanged)
