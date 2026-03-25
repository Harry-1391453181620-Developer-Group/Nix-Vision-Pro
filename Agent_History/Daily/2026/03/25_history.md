# Daily History - 2026-03-25

## Summary
- Replaced the fixed class metadata in `config.py` with dataset-driven runtime helpers that detect class names from `Dataset/` and derive `NUM_CLASSES` from the detected labels.
- Updated both training backends so `--class-count` is now an optional override instead of a required fixed input, while still validating that overrides cannot be smaller than the detected dataset classes.
- Updated both prediction backends and both inference GUIs to resolve class labels and model output size from the active dataset layout instead of static config lists.
- Updated the synthetic and Wikimedia dataset builders so they select their active class subset from `Dataset/` and only fall back to their built-in supported class sets when no dataset classes are available yet.
- Added regression tests for runtime class discovery and dataset-builder class resolution, and updated local documentation examples to stop hard-coding `--class-count 8`.

## Code Changes
- Added `detect_class_names`, `get_class_names`, and `get_num_classes` helpers in `config.py`, with support for padding placeholder labels when an explicit larger class count is requested.
- Kept dataset-builder safety behavior intact while replacing hard-coded active class lists with dataset-driven resolution functions:
  - `resolve_synthetic_class_names` in `build_synthetic_dataset.py`
  - `resolve_wikimedia_class_sources` in `build_wikimedia_dataset.py`
- Patched `backends/numpy/train_backend.py` and `backends/torch/train_backend.py` to compute `fallback_num_classes` dynamically, auto-detect the default class count from the dataset, and log the resolved runtime class list.
- Patched `backends/numpy/predict_backend.py`, `backends/torch/predict_backend.py`, `backends/numpy/gui_backend.py`, and `backends/torch/gui_backend.py` so they accept optional `--data-dir` / `--class-count` overrides and build models from the resolved runtime class list.
- Added `tests/test_runtime_class_config.py` to cover dataset-driven class-name detection, class-count padding, dataset-builder subset selection, and unsupported-class rejection.

## Validation
- Verified the touched Python modules compile successfully with `python -m compileall`.
- Ran `.venv\Scripts\python.exe -m pytest tests/test_runtime_class_config.py -q -p no:cacheprovider` and confirmed `5 passed`.
- Ran `.venv\Scripts\python.exe -m pytest tests/test_model.py tests/test_torch_model.py -q -p no:cacheprovider` and confirmed `4 passed`.
- Ran both training entrypoints with `--epochs 0` and no `--class-count`, and confirmed dataset-driven class auto-detection on the active `Dataset/` layout for both NumPy and PyTorch backends.

## Notes
- `pytest` under the project venv initially failed because the default Windows temp location was permission-restricted in this environment, so the new regression file was written to avoid `tmp_path` and keep tests inside the workspace.
- No git commit was created because this workspace is not a git repository.


## Checkpoint Resume Debugging
- Investigated why validation accuracy dropped from about `0.6` back toward `0.2` when training was restarted with only `--checkpoint`.
- Confirmed the root cause in both training backends: `--checkpoint` only saved weights, while loading only happened when `--init-from` was passed, so the next run started from scratch.
- Patched both `backends/torch/train_backend.py` and `backends/numpy/train_backend.py` to add `--resume / --no-resume` and automatically reuse the existing `--checkpoint` file when resuming is enabled and `--init-from` is not provided.
- Updated `README.md` and `Image_Identify_CNN.md` to document the corrected checkpoint semantics.
- Added `tests/test_train_resume.py` to lock the resume-path selection behavior.
- Added `checkpoint_resume_debug_report.md` to record the issue, evidence, fix, and remaining limitation.

## Checkpoint Resume Validation
- Verified the edited Python files compile with `python -m compileall`.
- Ran `\.venv\Scripts\python.exe -m pytest tests\test_train_resume.py tests\test_torch_model.py tests\test_model.py -q -p no:cacheprovider` and confirmed `8 passed`.
- Ran a PyTorch smoke test on a tiny local dataset and confirmed the trainer now prints `Resumed weights from checkpoint: ...` when `--checkpoint` already exists.

## Checkpoint Resume Notes
- Resume is currently weights-only; optimizer state, LR progress, and epoch counters are still not stored in checkpoints.
- No git commit was created because this workspace flow did not require git for the fix.

## Rollback And Git Setup
- Rolled back the checkpoint auto-resume change after confirming the training restart issue was caused by using `--checkpoint` without `--init-from`.
- Restored both training backends and both docs to their previous checkpoint semantics, and removed the temporary resume regression test and debug report files that were added for the investigation.
- Expanded `.gitignore` so the new git repository will not track `Dataset/`, workspace temp folders, pytest temp folders, or local log files.
- Initialized a git repository for the project and prepared it for a clean first commit on `main`.

## Rollback And Git Validation
- Verified the rollback removed the resume-specific code paths and temporary files.
- Reused the existing global git identity for repository setup.
