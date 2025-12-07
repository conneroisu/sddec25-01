# Tasks: Add Google Colab Notebook for EfficientViT Segmentation Training

## 1. Notebook Setup
- [x] 1.1 Create `training/train_efficientvit.ipynb` with Colab metadata
- [x] 1.2 Add setup cell for Colab GPU runtime check
- [x] 1.3 Add pip install cell for dependencies (torch, torchvision, opencv-python, datasets, etc.)

## 2. Dataset Loading
- [x] 2.1 Add cell to load OpenEDS dataset from HuggingFace (`Conner/openeds-precomputed`)
- [x] 2.2 Add dataset caching for Colab runtime persistence
- [x] 2.3 Display sample images from dataset for verification

## 3. Model and Training Components
- [x] 3.1 Port TinyEfficientViT encoder classes (TinyConvNorm, TinyPatchEmbedding, TinyCascadedGroupAttention, TinyLocalWindowAttention, TinyMLP, TinyEfficientVitBlock, TinyEfficientVitStage, TinyEfficientVitEncoder)
- [x] 3.2 Port TinySegmentationDecoder class
- [x] 3.3 Port TinyEfficientViTSeg model class
- [x] 3.4 Port CombinedLoss (cross-entropy + dice + surface loss)
- [x] 3.5 Port data augmentation classes (RandomHorizontalFlip, Gaussian_blur, Line_augment)
- [x] 3.6 Port IrisDataset class
- [x] 3.7 Port metric functions (compute_iou_tensors, finalize_iou, get_predictions, get_nparams)

## 4. Training Loop
- [x] 4.1 Port training configuration (hyperparameters, optimizer, scheduler)
- [x] 4.2 Port training epoch logic with progress bar (tqdm)
- [x] 4.3 Port validation loop and metrics computation
- [x] 4.4 Add checkpoint saving to Google Drive (optional) or local

## 5. Logging and Visualization
- [x] 5.1 Replace MLflow with inline matplotlib plots for training curves
- [x] 5.2 Port create_training_plots function for loss/IoU visualization
- [x] 5.3 Port create_prediction_visualization for sample outputs
- [x] 5.4 Display metrics summary at end of training

## 6. Export and Artifacts
- [x] 6.1 Add PyTorch model checkpoint save cell
- [x] 6.2 Add ONNX export cell
- [x] 6.3 Add cell to download trained model from Colab

## 7. Validation
- [ ] 7.1 Test notebook execution in Google Colab
- [ ] 7.2 Verify GPU utilization and training speed
- [ ] 7.3 Confirm model export works correctly
- [ ] 7.4 Verify model parameter count is <60k
