# Daily History - 2026-03-24

## Summary
- Investigated the earlier PyTorch CUDA failure in the custom Mamba path and stabilized that code path for the existing implementation.
- Rolled back the temporary mixed-device workaround so the model returned to full-GPU execution.
- Replaced the active architecture in both PyTorch and NumPy backends with a larger three-stage CNN + SE classifier.
- Removed Mamba from the active model paths and widened the classifier head to `FC(256)` with `Dropout(0.5)`.
- Updated documentation and focused tests to match the new architecture.

## Architecture Changes
- Replaced the previous active CNN + Mamba model in `backends/torch/model.py` with:
  - Stage 1: `Conv(3->32)`, `Conv(32->32)`, `SE(32)`, `MaxPool`
  - Stage 2: `Conv(32->64)`, `Conv(64->64)`, `SE(64)`, `MaxPool`
  - Stage 3: `Conv(64->128)`, `Conv(128->128)`, `SE(128)`, `MaxPool`
  - Head: `Flatten -> FC(256) -> ReLU -> Dropout(0.5) -> FC(num_classes)`
- Replaced the previous active NumPy CNN + Mamba model in `backends/numpy/model.py` with the same architecture.
- Added a NumPy `SqueezeExcitation` layer implementation in `backends/numpy/nn/layers.py` and exported it through `backends/numpy/nn/__init__.py`.

## Training Changes
- Updated the default classifier dropout in both training backends to `0.5`.
- Kept the existing optimizer, augmentation, scheduler, class-weighting, and early-stopping options intact.
- Preserved forgiving checkpoint loading behavior so partially compatible checkpoints can still initialize matching layers.

## Validation
- Verified that both active model modules import successfully.
- Updated the focused PyTorch tests to target the new CNN + SE architecture instead of the retired Mamba block.
- Planned validation target: `tests/test_model.py` and `tests/test_torch_model.py`.

## Notes
- The retired `Mamba` modules remain in the repository for now, but they are no longer used by the active model definitions.
- This architecture change invalidates strict compatibility with older full-model checkpoints because the parameter shapes and key sets changed substantially.
