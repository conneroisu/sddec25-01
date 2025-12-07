## 1. Implementation

- [x] 1.1 Create `train_ellipse_local.py` base file with imports (remove modal import)
- [x] 1.2 Add argparse CLI for configurable hyperparameters (epochs, batch_size, lr, data_dir, etc.)
- [x] 1.3 Convert dataset loading to use local cache directory (replace Modal Volume)
- [x] 1.4 Replace Modal secrets with environment variables for MLflow credentials
- [x] 1.5 Remove Modal decorators and move all code to main execution block
- [x] 1.6 Add device detection (CUDA/CPU) with user override option
- [x] 1.7 Add local model checkpoint saving to configurable output directory

## 2. Validation

- [x] 2.1 Verify script runs without Modal installed (syntax check passed, no modal imports)
- [ ] 2.2 Test dataset download and caching on local filesystem
- [ ] 2.3 Verify training loop executes correctly on GPU/CPU
- [ ] 2.4 Confirm MLflow logging works with environment variable credentials
- [ ] 2.5 Validate model checkpoint saving and loading
