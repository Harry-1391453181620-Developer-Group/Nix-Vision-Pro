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

## Follow-up update

- Changed default temporary backbone freeze patience from `5` to `8` via the new `--freeze-patience` flag.
- Changed temporary freeze behavior from "until next phase" to a timed window controlled by `--freeze-epoch-num`, with default `10` epochs.
- Added `--after-unfreeze-lr-change` as an additive LR decrement applied after unfreeze only when the resulting LR stays positive and does not go below the next phase start LR.
- Increased training rotation range from `+-10°` to `+-12°` in the shared augmentation policy.
- Updated `Image_Identify_CNN.md`, `README.md`, `CONTRIBUTING.md`, and the design note to document the new behavior.
- Preserved the existing numeric values already present in the example training commands inside `Image_Identify_CNN.md` and only appended the new flags there.

## Second follow-up update

- Replaced the old post-unfreeze phase-base-LR rewrite with a cumulative effective-LR offset policy.
- Effective LR is now computed as `max(scheduled_lr - phase_lr_offset, min_lr)` so the optimizer never drops below the scheduler floor.
- Post-unfreeze deductions now use the current effective LR, not the phase base LR, and only apply when the LR is still meaningfully above the floor.
- Added a small epsilon guard for next-phase LR comparisons and capped the cumulative offset so it never undercuts the next phase start LR.
- Updated focused tests and documentation to match the refined cumulative deduction policy.
