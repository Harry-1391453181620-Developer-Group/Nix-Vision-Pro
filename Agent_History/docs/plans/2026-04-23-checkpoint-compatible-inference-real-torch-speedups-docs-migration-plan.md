# Checkpoint-Compatible Inference, Real Torch Speedups, and Docs Migration

## Summary
- Fix the current GUI/model-load failure by making inference reconstruct the model from checkpoint architecture instead of current defaults.
- Keep `Dataset/` and `best_train_commands.txt` unchanged.
- Refactor the PyTorch training path so the speed work targets real bottlenecks: Windows-safe `DataLoader` usage, stateless workers, contiguous pinned CPU tensors, non-blocking H2D transfer for both inputs and labels, GPU MixUp/CutMix, correct `channels_last` usage, runtime-gated AMP, and compile benchmarking with warmup and synchronization.
- Move root `docs/` to `Agent_History/docs/` and update active references.

## Implementation Changes

### Checkpoint and inference
- Extend new torch and numpy checkpoints to save `num_classes`, `width_scale`, `stage2_channels`, `input_size`, and `class_names`.
- Add legacy fallback inference that derives architecture from weight shapes when metadata is missing: torch from `conv3.weight` and `fc2.weight`, numpy from `conv3.W` and `fc2.W`.
- Update both torch and numpy GUI/predict backends so checkpoint architecture becomes authoritative once weights are loaded; conflicting CLI overrides fail clearly.
- Resolve labels as checkpoint `class_names` first, dataset-derived names second when counts match, numeric placeholders otherwise.

### Torch data pipeline
- Replace the current hand-rolled batch loop with a dataset plus `torch.utils.data.DataLoader`.
- Keep the torch dataset stateless for Windows multiprocessing: only immutable/read-only members such as path lists, config values, and class metadata; no mutable caches, worker-local rolling state, or shared RNG state stored on the dataset instance.
- Preserve Windows multiprocessing safety by keeping training entrypoints under `if __name__ == "__main__": main()` and not introducing any new worker-spawning path outside that guard.
- In streaming mode, move PIL load, resize/normalize, and per-image RandAugment/cutout into dataset workers so CPU-heavy work is actually parallelized.
- Keep validation clean: no augmentation and no batch mixing.
- Make dataset/collate output CPU tensors, not NumPy arrays.
- Require collate outputs to be contiguous and typed for transfer efficiency:
  - images: contiguous `float32`
  - labels / class ids: contiguous `int64`
- Use `pin_memory=True` on CUDA runs and transfer both inputs and labels with `to(device, non_blocking=True)`.
- Add `--num-workers`; default to `4` for streaming mode and `0` for preloaded mode on Windows. Enable `persistent_workers` and `prefetch_factor=2` only when workers > 0.
- Add `worker_init_fn` seeding so worker-local augmentation randomness is deterministic but non-duplicated:
  - seed from base run seed + worker id
  - initialize `numpy` and `random` in each worker

### GPU MixUp / CutMix
- Remove CPU-side batch mixing from the torch training loop.
- Reimplement the torch backend’s MixUp/CutMix directly on device tensors after H2D transfer.
- Use only torch RNG on the target device for batch mixing:
  - `lam` from a torch Beta distribution on device
  - `randperm` from `torch.randperm(..., device=device)`
  - CutMix box randomness from torch device RNG as well
- Ensure `lam` matches the active tensor dtype/device:
  - `lam.to(device=device, dtype=x.dtype)`
- Do not use NumPy RNG anywhere in the torch backend’s batch-mixing path.
- Keep current semantics unchanged: one batch-level gate, then CutMix vs MixUp routing, label smoothing disabled on mixed batches, focal loss disabled on mixed batches.

### Layout, kernels, and AMP
- Normalize the optimized torch training path to NCHW tensors before forward.
- Put both the live model and EMA model on device with channels-last in one step:
  - `model.to(device=device, memory_format=torch.channels_last)`
- Put training and eval input tensors into `torch.channels_last` after device transfer as well.
- Add a fast path in `TorchCNN.forward()` for already-NCHW tensors and avoid repeated `.contiguous()` in the optimized path.
- Legacy NHWC callers remain supported, but training should not depend on NHWC reformatting inside `forward()`.
- Enable `torch.backends.cudnn.benchmark = True` only while spatial input shape is fixed; if a future variable-shape mode such as resize jitter or multi-scale training is added, auto-disable it.
- Enable `torch.set_float32_matmul_precision("high")` on supported torch versions.
- Add `--amp-mode {auto,on,off}`:
  - `auto`: on CUDA, use BF16 when `torch.cuda.is_bf16_supported()` is true, otherwise FP16 with `GradScaler`; on CPU disable AMP
  - `on`: require AMP on the best supported CUDA dtype and fail if unsupported
  - `off`: keep full precision
- Use `GradScaler` only for FP16 AMP. Do not use `GradScaler` for BF16.

### Compile policy
- Add `--compile-mode {auto,on,off}`.
- `auto`: only try compile on CUDA runs.
- Benchmark compile with warmup protection and synchronized timing:
  - eager warmup: 5 steps
  - eager measure: 10 steps median
  - compile warmup: 5 steps
  - compile measure: 10 steps median
- Surround each measured step with `torch.cuda.synchronize()` so timing reflects real GPU step latency rather than async launch time.
- Use synchronized train-step timing, not raw forward timing.
- Keep compiled mode only if median compiled train-step time improves by at least 5%; otherwise revert to eager and log that compile was disabled due to no benefit or slowdown.
- If compile initialization or execution fails for any reason, log the concrete reason once and continue in eager mode.
- `on`: require compile success and keep it enabled once initialized.
- `off`: skip compile entirely.

### Docs migration
- Move `docs/plans/*.md` to `Agent_History/docs/plans/` and remove the root `docs/` folder afterward.
- Update `CONTRIBUTING.md` and remaining markdown/history references that still point to `docs/plans/`.
- Refresh docs that still mention stale class counts or outdated inference width assumptions.

## Public Interface Changes
- New training flags:
  - `--num-workers`
  - `--amp-mode {auto,on,off}`
  - `--compile-mode {auto,on,off}`
- New checkpoint metadata fields:
  - `num_classes`
  - `width_scale`
  - `stage2_channels`
  - `input_size`
  - `class_names`
- Inference behavior change:
  - once weights are loaded, checkpoint architecture wins over current default width/class settings

## Test Plan
- Reproduce the current torch GUI/predict checkpoint-load failure with `checkpoints/best_torch_model.pt`, then verify it loads without manually passing `--model-width-scale`.
- Add unit tests for checkpoint architecture recovery with and without metadata for both backends.
- Add regression tests for CLI override conflicts against checkpoint metadata.
- Add trainer tests for:
  - stateless dataset construction and worker-safe loading assumptions
  - collate output dtype and contiguity
  - CUDA pin-memory / non-blocking transfer path selection for both inputs and labels
  - GPU MixUp/CutMix correctness against current policy
  - AMP mode selection using runtime capability checks
  - FP16-only `GradScaler` behavior
  - compile auto-fallback on exception
  - compile auto-disable when benchmarked median step time is not better
  - synchronized compile benchmark timing path
  - cuDNN benchmark gating remaining on only for fixed-size inputs
- Run a CUDA smoke run on a tiny subset to compare baseline vs optimized median train-step time and confirm training, validation, EMA save, and checkpoint load still work.
- Verify no active repo references remain to `docs/plans/` after the move.

## Assumptions and Defaults
- `Dataset/` and `best_train_commands.txt` stay untouched.
- The performance goal is reduced GPU idle time and lower median train-step / epoch wall time on the current RTX 5070 setup, not a guaranteed fixed multiplier.
- NumPy backend gets checkpoint/inference compatibility fixes, but the performance refactor is limited to the PyTorch training backend.
- Torch-side training semantics should stay aligned with current augmentation/mixup/focal/EMA behavior even if the internal implementation becomes backend-specific for performance.

## Status
- Implemented.
