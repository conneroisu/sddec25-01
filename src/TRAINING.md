# Training TinyEfficientViTSeg

This document explains how to train the TinyEfficientViTSeg model locally.

## Requirements

The training script requires:
- PyTorch (with CUDA support recommended)
- HuggingFace datasets
- torchvision
- tqdm
- PIL
- numpy

These should already be installed if you've set up the virtual environment.

## Quick Start

Basic training with default parameters:

```bash
cd /home/connerohnesorge/Documents/001Repos/sddec25-01/src
.venv/bin/python train.py
```

## Command Line Arguments

```bash
.venv/bin/python train.py --help
```

Available options:

- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 15)
- `--lr`: Initial learning rate (default: 0.001)
- `--seed`: Random seed for reproducibility (default: 42)
- `--num-workers`: Number of dataloader workers (default: 4)
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints)

## Example Training Commands

### Quick test run (2 epochs, small batch)
```bash
.venv/bin/python train.py --epochs 2 --batch-size 8
```

### Full training run (default 15 epochs)
```bash
.venv/bin/python train.py --epochs 15 --batch-size 32
```

### Extended training (100+ epochs for alpha decay)
```bash
.venv/bin/python train.py --epochs 150 --batch-size 32 --lr 0.001
```

### CPU training (slower, no GPU)
```bash
# Will automatically detect and use CPU if CUDA not available
.venv/bin/python train.py --batch-size 16 --num-workers 2
```

## Output

The training script will create a `checkpoints/` directory containing:

1. **best_model.pth** - PyTorch checkpoint with best validation IoU
   - Contains: model weights, optimizer state, scheduler state, metrics
   
2. **checkpoint_epoch_N.pth** - Periodic checkpoints every 5 epochs

## Training Details

### Dataset
- **Source**: HuggingFace `Conner/openeds-precomputed`
- **Size**: ~3GB (downloads on first run, cached thereafter)
- **Image size**: 640x400 grayscale
- **Classes**: 2 (background=0, pupil=1)
- **Splits**: train, validation

### Data Processing
- **Minimal preprocessing**: Only normalization to [-1, 1]
- **No augmentation**: Raw images used directly
- **Precomputed features**: spatial weights, distance maps included

### Model Architecture
- **Parameters**: ~12,798 trainable parameters
- **Architecture**: TinyEfficientViTSeg (encoder-decoder with attention)
- **Input**: 640x400 grayscale images
- **Output**: 640x400 segmentation masks (2 classes)

### Training Configuration
- **Loss**: CombinedLoss (weighted CE + dice + surface)
- **Alpha scheduling**: Decays from 1→0 over 125 epochs
  - Alpha controls dice vs surface loss balance
- **Optimizer**: Adam with initial LR 1e-3
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Mixed precision**: Enabled automatically on GPU (AMP)

### Expected Performance
- **Validation mIoU**: ~0.95+ after 15 epochs
- **Training time**: 
  - GPU (L4/A100): ~5-10 min per epoch
  - CPU: ~1-2 hours per epoch

### Memory Requirements
- **GPU**: 4-8GB VRAM recommended (batch_size=32)
- **CPU RAM**: 8-16GB recommended

## Monitoring Training

The training script prints metrics each epoch:

```
Epoch 1/15
  Train Loss: 0.2345 | Valid Loss: 0.1987
  Train IoU:  0.9245 | Valid IoU:  0.9321
  CE Loss:    0.1234 | 0.1098
  Dice Loss:  0.0567 | 0.0543
  Surf Loss:  0.0544 | 0.0346
  LR: 0.001000 | Alpha: 0.9920
```

Watch for:
- **Decreasing loss** on both train and validation
- **Increasing IoU** (Intersection over Union)
- **LR changes** when validation plateaus
- **Best model saves** when validation IoU improves

## Loading Trained Models

### Load PyTorch checkpoint
```python
import torch
from model import TinyEfficientViTSeg

model = TinyEfficientViTSeg(...)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Troubleshooting

### Out of memory errors
- Reduce `--batch-size` to 16 or 8
- Reduce `--num-workers` to 2 or 0

### Slow training on CPU
- Use smaller batch size (8-16)
- Consider using GPU (much faster)
- Reduce num_workers to match CPU cores

### Dataset download issues
- Check internet connection
- HuggingFace may be rate-limiting (wait and retry)
- Dataset is cached after first download

### Import errors
- Ensure virtual environment is activated
- Check that model.py is in the same directory
- Verify all dependencies are installed

## Notes

- First run downloads ~3GB dataset (cached for future runs)
- Training is fully reproducible with `--seed`
- Checkpoints can be used to resume training
