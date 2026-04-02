# 2026-03 Monthly Summary

## Scope

This summary covers the recorded project work from 2026-03-23 through 2026-03-27, plus the main repository changes visible in the March commit history.

## Main Outcomes

- Reorganized the project into explicit `torch` and `numpy` backends while keeping top-level compatibility entrypoints.
- Made the PyTorch backend the default runtime path without deleting the NumPy backend.
- Replaced the older active CNN+Mamba architecture with the current three-stage CNN + SE classifier.
- Switched runtime class metadata from fixed constants to dataset-driven discovery from `Dataset/`.
- Added shared training-policy infrastructure for augmentation, phase scheduling, timed backbone freeze, width scaling, MixUp, focal loss, CutMix, RandAugment, and EMA.
- Added structured checkpoint metadata and compatible legacy checkpoint loading.
- Initialized git for the repository and established the current commit history.

## Detailed Progress

### 2026-03-23

- Repaired the synthetic and Wikimedia dataset builder scripts at that time.
- Added training improvements to the NumPy path, including BatchNorm support and stronger trainer controls.
- Split the repository into `backends/torch/` and `backends/numpy/`.
- Updated the environment and docs around the Python 3.14 project `.venv`.

### 2026-03-24

- Stabilized the earlier CUDA-related Mamba investigation.
- Replaced the active runtime architecture in both backends with the current CNN + SE classifier.
- Kept Mamba in the repository only as inactive legacy code at that point.

### 2026-03-25

- Added dataset-driven class discovery via `config.py`.
- Updated both training, prediction, and GUI backends to resolve class names from the actual dataset layout.
- Added regression coverage for runtime class discovery.
- Investigated checkpoint resume behavior, then rolled back the temporary auto-resume patch after confirming the root cause was missing `--init-from`.
- Initialized git and tightened `.gitignore`.

### 2026-03-26

- Added shared training utilities for deterministic augmentation and per-phase LR scheduling.
- Added timed backbone freeze and post-unfreeze LR deduction behavior in both backends.
- Updated docs and focused tests for the new training controls.

### 2026-03-27

- Added configurable augmentation strengths, random erasing, MixUp, focal loss, and width scaling across both backends.
- Repaired a torch trainer regression caused by missing helper functions.
- Enabled MixUp by default and added `--mixup-alpha`.
- Documented the approved designs in `docs/plans/` and committed the work.

## Documentation And Process

- `README.md`, `Image_Identify_CNN.md`, and `CONTRIBUTING.md` were repeatedly updated to match runtime behavior.
- Daily histories were written for the active workdays in March.
- Design notes were added under `docs/plans/` for major approved behavior changes.

## Validation Themes

- Focused compile checks were used for touched modules.
- Regression coverage was added and expanded for runtime class discovery, training policy, checkpoint behavior, and model compatibility.
- Both NumPy and PyTorch backends were kept under test during the architecture and training-policy changes.

## Closing Notes

- March was the month where the repository moved from a single older path into the current dual-backend, dataset-driven, git-tracked training project.
- Some legacy code introduced earlier in March remained in the repository after month end and was cleaned up later in April.
