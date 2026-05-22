# 2026-03-27 History

## Request

Added configurable preprocessing strengths, random erasing, MixUp, focal loss, and width-scaled model construction across both training backends. Updated docs, tests, and checkpoint-loading compatibility, then committed the work.

## Work completed

- Wrote the approved design note to `Agent_History/docs/plans/2026-03-27-dual-backend-focal-mixup-design.md`.
- Updated `utils/training.py` so both backends now share:
  - validated augmentation strengths for rotation, brightness, contrast, and saturation
  - random erasing with `10%-20%` area sampling and per-image mean fill
  - MixUp with `Beta(0.2, 0.2)` and per-batch activation probability
  - width-scale validation for the resized stage-2 block
- Updated `backends/torch/model.py` and `backends/numpy/model.py` to support `width_scale`, with default stage-2 width `48` from `0.75 * 64`.
- Updated `backends/torch/train_backend.py` and `backends/numpy/train_backend.py` to support:
  - `--rotation`
  - `--brightness`
  - `--contrast`
  - `--saturation`
  - `--mixup / --no-mixup`
  - `--mixup-prob`
  - `--focal-loss / --no-focal-loss`
  - `--focal-gamma`
  - `--focal-alpha {auto,none}`
  - `--model-width-scale`
- Replaced the old balance-sampling path with focal-loss handling in both trainers.
- Added automatic run-level fallback from focal loss to cross entropy when MixUp is enabled.
- Updated `backends/numpy/nn/losses.py` with focal-loss forward and backward implementations.
- Updated both predict backends and both GUI backends to accept `--model-width-scale` so inference can match the training-time checkpoint shape.
- Added focused tests for:
  - focal-loss equivalence at `gamma=0`
  - focal-loss gradient shape
  - MixUp soft-label behavior
  - random-erasing mean fill
  - width-scaled stage-2 channel counts
  - forgiving checkpoint loading when width-scaled tensors differ
- Updated `Image_Identify_CNN.md` and `README.md` to document the new behavior and example commands.

## Validation

- `python -m py_compile utils	raining.py backends	orch\model.py backends
umpy\model.py backends	orch	rain_backend.py backends
umpy	rain_backend.py backends	orch\predict_backend.py backends
umpy\predict_backend.py backends	orch\gui_backend.py backends
umpy\gui_backend.py backends
umpy
n\losses.py tests	est_losses.py tests	est_mixup_focal_width.py`
- `.\.venv\Scripts\python.exe -m pytest tests	est_losses.py tests	est_training_policies.py tests	est_model.py tests	est_torch_model.py tests	est_mixup_focal_width.py -q`

## Notes

- The first test attempt used the wrong interpreter and failed because `pytest` was not installed there. The rerun used the project virtualenv and passed.
- Pytest still reports a cache-write warning in this environment because `.pytest_cache` cannot create its nested cache directory cleanly here.
- Passive security review: no new high-severity security issues were introduced in the added CLI, checkpoint-loading, or loss-selection paths. The checkpoint readers still use explicit extension checks and NumPy loading remains `allow_pickle=False`.


## Torch Trainer Repair

- Investigated reported errors in `backends/torch/train_backend.py` using direct module inspection instead of only `py_compile`.
- Found that a previous block replacement had accidentally removed these required helper functions from the file:
  - `stable_partition_index`
  - `choose_partition`
  - `_resolve_device`
  - `_set_optimizer_lr`
  - `_to_tensor_batch`
  - `_load_batch`
- Restored the missing helper block without changing the newer focal-loss, MixUp, or width-scale logic.
- Revalidated with:
  - `.\.venv\Scripts\python.exe -c "import backends.torch.train_backend as m; ..."`
  - `.\.venv\Scripts\python.exe -m py_compile backends	orch	rain_backend.py`
  - `.\.venv\Scripts\python.exe -m pytest tests	est_training_policies.py tests	est_torch_model.py tests	est_mixup_focal_width.py -q`


## Request

Enabled MixUp by default in both training backends, added `--mixup-alpha` with default `0.2`, kept `--mixup-prob 0.5`, updated docs, and prepared a commit without touching `best_train_commands.txt`.

## Work completed

- Added `validate_mixup_alpha()` to `utils/training.py` so invalid Beta parameters fail fast before training starts.
- Updated `backends/torch/train_backend.py` and `backends/numpy/train_backend.py` so `--mixup` now defaults to enabled, `--mixup-alpha` is exposed on the CLI, and the validated alpha value drives the existing MixUp helper.
- Extended `tests/test_mixup_focal_width.py` to cover valid and invalid `mixup_alpha` values.
- Updated `README.md` and `Image_Identify_CNN.md` so the documented defaults and example commands match the trainer behavior.
- Wrote `Agent_History/docs/plans/2026-03-27-default-mixup-alpha-design.md` to capture the approved design delta for this follow-up change.

## Validation

- `.\.venv\Scripts\python.exe -m py_compile utils\training.py backends\torch\train_backend.py backends\numpy\train_backend.py tests\test_mixup_focal_width.py`
- `.\.venv\Scripts\python.exe -m pytest tests\test_mixup_focal_width.py tests\test_training_policies.py -q`

## Notes

- Passive security review: this change only adjusts safe numeric CLI validation and loss-policy selection; it does not expand file, network, or deserialization attack surface.
- Test warning: pytest still emitted the existing `.pytest_cache` directory warning in this environment.
