# Training Spec Delta

## REMOVED Requirements

### Requirement: ONNX Export
**Reason**: Replacing ONNX export with PyTorch checkpoint format for better compatibility with PyTorch workflows and consistency with EfficientViT training script.
**Migration**: Existing ONNX artifacts in MLflow remain valid; new training runs will produce `.pt` files instead of `.onnx` files.

## ADDED Requirements

### Requirement: Ellipse Model PyTorch Checkpoint Export
The system SHALL export the trained ellipse regression model to PyTorch checkpoint format compatible with edge deployment and fine-tuning.

#### Scenario: PyTorch checkpoint export
- **WHEN** training completes with best validation mIoU
- **THEN** best model SHALL be saved to `best_ellipse_model.pt`
- **AND** checkpoint SHALL contain model `state_dict()`
- **AND** model SHALL be in contiguous memory format for portability

#### Scenario: Epoch checkpoints
- **WHEN** training reaches checkpoint epochs (every 10 epochs or final epoch)
- **THEN** checkpoint SHALL be saved as `ellipse_model_epoch_{n}.pt`
- **AND** all checkpoints SHALL be uploaded to MLflow as artifacts

#### Scenario: Compiled model handling
- **WHEN** model is wrapped with `torch.compile()`
- **THEN** export function SHALL unwrap to `_orig_mod` before saving
- **AND** checkpoint SHALL contain unwrapped model weights
