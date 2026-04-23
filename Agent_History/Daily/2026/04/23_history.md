# 2026-04-23 History

## Request

Implemented checkpoint-compatible inference, the torch training runtime refactor, and the docs migration to `Agent_History/docs/`, without touching `Dataset/` or `best_train_commands.txt`.

## Work completed

- Updated both model backends so new checkpoints save runtime metadata for:
  - `num_classes`
  - `width_scale`
  - `stage2_channels`
  - `input_size`
  - `class_names`
- Added legacy checkpoint architecture recovery in both backends by inferring class count and stage-2 width from saved parameter shapes when metadata is missing.
- Updated both predict backends and both GUI backends so checkpoint architecture now wins when weights are loaded, and conflicting `--class-count` or `--model-width-scale` overrides fail clearly.
- Added `config.resolve_runtime_class_names()` so inference resolves labels in this order:
  - checkpoint class names
  - dataset-derived names when counts match
  - numeric placeholders
- Reworked `backends/torch/train_backend.py` to use:
  - `DataLoader`
  - worker-safe stateless datasets
  - contiguous tensor collation
  - `pin_memory` and non-blocking transfer on CUDA
  - GPU MixUp/CutMix
  - `channels_last`
  - AMP with BF16/FP16 gating
  - synchronized `torch.compile` benchmarking with auto-disable
- Patched `backends/numpy/train_backend.py` so new NumPy checkpoints also save `class_names`.
- Expanded regression coverage for:
  - runtime class-name resolution
  - structured and legacy checkpoint runtime config recovery
  - checkpoint override conflict handling
  - collate contiguity
  - worker RNG seeding
  - torch AMP/scaler policy
  - torch batch-mix simplex preservation
- Updated `README.md`, `Image_Identify_CNN.md`, `CONTRIBUTING.md`, and history references so active docs now point to `Agent_History/docs/plans/`.

## Validation

- `python -m py_compile config.py backends\torch\model.py backends\numpy\model.py backends\torch\predict_backend.py backends\torch\gui_backend.py backends\numpy\predict_backend.py backends\numpy\gui_backend.py backends\torch\train_backend.py`
- `python -m py_compile backends\numpy\train_backend.py tests\test_runtime_class_config.py tests\test_torch_model.py tests\test_model.py tests\test_training_policies.py`
- Direct checkpoint smoke checks:
  - resolved `checkpoints/best_torch_model.pt` to `num_classes=13`, `stage2_channels=115`, `width_scale=1.796875`
  - reconstructed `TorchCNN` from that config and loaded weights successfully
- Torch training smoke run on a temporary dataset:
  - `python train.py --backend torch --data-dir .worktmp\train_smoke_dataset --epochs 1 --batch-size 2 --device cpu --streaming --num-workers 0 --amp-mode off --compile-mode off --no-ema --mixup --mixup-alpha 0.2 --mixup-prob 1.0 --cutmix-ratio 0.5 --checkpoint .worktmp\smoke_torch.pt --val-split 0.25 --phase-count 1 --lr 0.001 --no-early-stop`

## Notes

- `pytest` is not installed in the available interpreter in this session, so the added regression tests were compile-checked but not executed under `pytest`.
- The torch checkpoint smoke validation confirmed the original GUI load failure was fixed at the checkpoint-architecture reconstruction layer.
