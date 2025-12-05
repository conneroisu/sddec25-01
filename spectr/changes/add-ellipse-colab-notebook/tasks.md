# Tasks: Add Google Colab Notebook for Ellipse Regression Training

## 1. Notebook Setup
- [ ] 1.1 Create `training/train_ellipse.ipynb` with Colab metadata
- [ ] 1.2 Add setup cell for Colab GPU runtime check
- [ ] 1.3 Add pip install cell for dependencies (torch, torchvision, opencv-python, datasets, etc.)

## 2. Dataset Loading
- [ ] 2.1 Add cell to load OpenEDS dataset from HuggingFace
- [ ] 2.2 Add dataset caching for Colab runtime persistence
- [ ] 2.3 Display sample images from dataset for verification

## 3. Model and Training Components
- [ ] 3.1 Port `DownBlock` and `EllipseRegressionNet` model classes
- [ ] 3.2 Port `EllipseRegressionLoss` loss function
- [ ] 3.3 Port ellipse parameter extraction and normalization utilities
- [ ] 3.4 Port data augmentation classes (RandomHorizontalFlip, Gaussian_blur, Line_augment)
- [ ] 3.5 Port `IrisDataset` class

## 4. Training Loop
- [ ] 4.1 Port training configuration (hyperparameters, optimizer, scheduler)
- [ ] 4.2 Port training epoch logic with progress bar
- [ ] 4.3 Port validation loop and metrics computation
- [ ] 4.4 Add checkpoint saving to Google Drive (optional) or local

## 5. Logging and Visualization
- [ ] 5.1 Replace MLflow with TensorBoard or inline matplotlib plots
- [ ] 5.2 Add training curves visualization cell
- [ ] 5.3 Add prediction visualization cell for sample outputs
- [ ] 5.4 Display metrics summary at end of training

## 6. Export and Artifacts
- [ ] 6.1 Add ONNX export cell
- [ ] 6.2 Add cell to download trained model from Colab

## 7. Validation
- [ ] 7.1 Test notebook execution in Google Colab
- [ ] 7.2 Verify GPU utilization and training speed
- [ ] 7.3 Confirm model export works correctly
