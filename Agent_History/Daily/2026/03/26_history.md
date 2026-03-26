# 2026-03-26 History

## Request

Added controlled augmentation, temporary backbone freeze, and multiphase LR training to both backends.

## Work completed

- Wrote the approved design note to `docs/plans/2026-03-26-phase-freeze-augment-design.md`.
- Added `utils/training.py` so both backends share:
  - deterministic augmentation order
  - reflection-padded centered rotation
  - bounded color jitter
  - phase splitting with `np.array_split`
  - per-phase warmup and LR scheduling helpers
- Updated `backends/torch/model.py` with explicit backbone/head iteration helpers for freeze control.
- Updated `backends/numpy/model.py` with backbone freeze state, BN freeze handling, and active parameter selection helpers.
- Fixed `backends/numpy/nn/layers.py` so BatchNorm backward remains correct when running stats are frozen and affine parameters are optionally trainable.
- Reworked `backends/torch/train_backend.py` and `backends/numpy/train_backend.py` to support:
  - `--phase-count`
  - phase LR lists via `--lr`
  - `--warmup-epochs`
  - monotonic LR validation
  - cosine restarts per phase
  - temporary backbone freeze after 5 `val_acc` plateau epochs
  - automatic unfreeze at the next phase boundary
  - optional `--freeze-bn-affine false`
- Added `tests/test_training_policies.py` to cover:
  - augmentation determinism
  - phase schedule validation
  - NumPy BN running-stat freeze behavior
  - PyTorch freeze -> unfreeze -> freeze transition behavior
- Updated `README.md`, `Image_Identify_CNN.md`, and `CONTRIBUTING.md` to document the new runtime behavior.

## Validation

- `python -m py_compile utils\training.py backends\torch\model.py backends\torch\train_backend.py backends\numpy\model.py backends\numpy\train_backend.py tests\test_training_policies.py`
- `.\.venv\Scripts\python.exe -m pytest tests\test_training_policies.py tests\test_model.py tests\test_torch_model.py -q`

## Notes

- Pytest passed with one cache write warning because `.pytest_cache` is read-restricted in this environment.
- The new cosine schedule intentionally restarts inside each training phase.
