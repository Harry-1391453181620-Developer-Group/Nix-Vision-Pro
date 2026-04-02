# 2026-04-02 History

## Request

Added shared RandAugment, CutMix routing, EMA, and structured checkpoint metadata across both training backends. Updated tests, docs, and command guidance, then prepared the work for validation and commit.

## Work completed

- Wrote the approved design note to `docs/plans/2026-04-02-cutmix-randaugment-ema-design.md`.
- Updated `utils/training.py` so both backends now share:
  - RandAugment-style sampled augmentation ops
  - CutMix with area-corrected label mixing
  - single-gate MixUp/CutMix routing
  - EMA shadow-model handling over full state dicts, including buffers
  - EMA phase warmup and a small mixed-batch decay adjustment
- Updated `backends/torch/train_backend.py` and `backends/numpy/train_backend.py` to add:
  - `--cutmix-ratio`
  - `--ema / --no-ema`
  - `--ema-decay`
  - `--mixup-prob` default `0.4`
  - EMA validation and best-checkpoint evaluation
  - batch-level focal-loss fallback for mixed batches only
- Updated `backends/torch/model.py` and `backends/numpy/model.py` so new checkpoints save structured `model + meta` payloads while still loading legacy checkpoints.
- Refreshed tests to cover CutMix routing, EMA state tracking, structured checkpoint metadata, and legacy checkpoint compatibility.
- Updated `README.md`, `Image_Identify_CNN.md`, and `best_train_commands.txt` to reflect the new training policy.

## Validation

- `.\.venv\Scripts\python.exe -m py_compile utils\training.py backends\torch\model.py backends\numpy\model.py backends\torch\train_backend.py backends\numpy\train_backend.py tests\test_training_policies.py tests\test_mixup_focal_width.py tests\test_torch_model.py tests\test_model.py`
- `.\.venv\Scripts\python.exe -m pytest tests\test_training_policies.py tests\test_mixup_focal_width.py tests\test_torch_model.py tests\test_model.py -q -p no:cacheprovider`
- First pytest pass exposed a test-only dropout-mode mismatch in the legacy torch checkpoint regression; the test was corrected and the rerun passed with `22 passed`.

## Notes

- Passive security review: no new high-severity security issue was introduced. NumPy checkpoint loading still uses `allow_pickle=False`, and the PyTorch checkpoint reader prefers `weights_only=True` before falling back for compatibility.
- `agent-memory-mcp` remains unavailable in this environment, so the design and implementation record were written to repository files instead.
