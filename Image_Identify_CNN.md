# Image_Identify_CNN Training Guide

The project uses `PyTorch` as the default backend and keeps the original `NumPy` structure intact. The active architecture in both backends is a three-stage CNN + SE classifier with a flattened dense head.

## Current Runtime

- Python version: `3.14`
- Recommended interpreter: `./.venv/Scripts/python.exe`
- Default backend: `torch`
- Legacy backend: `numpy`
- Active dataset source: real images stored under `Dataset/` by default

## Active Architecture

Both backends now use:
- Stage 1: `Conv(3->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> SE(32) -> MaxPool`
- Stage 2: `Conv(32->round(64*scale)) -> BN -> ReLU -> Conv(round(64*scale)->round(64*scale)) -> BN -> ReLU -> SE(round(64*scale)) -> MaxPool`
- Stage 3: `Conv(round(64*scale)->128) -> BN -> ReLU -> Conv(128->128) -> BN -> ReLU -> SE(128) -> MaxPool`
- Head: `Flatten -> FC(256) -> ReLU -> Dropout(0.5) -> FC(num_classes)`

Width scaling is controlled by `--model-width-scale`.

- default `0.75` gives stage-2 width `48`
- inference now reconstructs width and class count from checkpoint metadata or legacy weight shapes, so explicit width overrides are mainly for training or weight-free model construction

The old Mamba implementation has been removed because it was no longer part of the active model path.

## Environment Setup

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset Layout

```text
Dataset/
  class_name_1/
  class_name_2/
  ...
```

Each class should be a directory. Training auto-detects the class count from `Dataset/`; use `--class-count` only when you need to override the detected output size.
Checkpoint-backed inference now prefers the saved checkpoint architecture and only falls back to dataset-derived labels when checkpoint class names are absent or incomplete.

The repository no longer ships synthetic or Wikimedia dataset-builder scripts. Current training and inference operate on the real image dataset only.

## Global Backend Switch

```text
--backend {torch,numpy}
```

## Training Behavior

### RandAugment

When `--augment` is enabled, both backends apply the same RandAugment-style policy from `utils/training.py`.

Policy details:
- each training image samples `2` ops with replacement from a shared op pool
- the op pool is: `rotate`, `brightness`, `contrast`, `saturation`, `sharpness`, `posterize`, `solarize`, `autocontrast`, `equalize`, `invert`, and `cutout`
- `--rotation x` still controls the maximum absolute rotation magnitude
- `--brightness x`, `--contrast x`, and `--saturation x` still control the photometric strength limits
- `cutout` keeps the mean-fill random erasing behavior
- validation never calls this augmentation path

Validation rules:
- `0 <= rotation < 180`
- `0 <= brightness <= 1`
- `0 <= contrast <= 1`
- `0 <= saturation <= 1`

### MixUp, CutMix, and Focal Loss

New arguments:
- `--mixup / --no-mixup`
- `--mixup-alpha`
- `--mixup-prob`
- `--cutmix-ratio`
- `--focal-loss / --no-focal-loss`
- `--focal-gamma`
- `--focal-alpha {auto,none}`

Policy details:
- batch mixing defaults to enabled
- `--mixup-prob` now defaults to `0.4`
- when a batch is selected for mixing, `--cutmix-ratio` decides between CutMix and MixUp and defaults to `0.5`
- CutMix recomputes label `lam` from the actual pasted patch area after clipping
- label smoothing becomes `0` on mixed batches to avoid double regularization
- focal loss stays enabled only for non-mixed training batches
- in the torch backend, MixUp and CutMix now run on device tensors after transfer instead of on CPU NumPy batches
- validation is always clean:
  - MixUp off
  - CutMix off

### Torch Runtime Optimization

Torch training now uses:
- `DataLoader` instead of a hand-rolled batch loader
- worker-local RNG seeding for augmentation when `num_workers > 0`
- contiguous CPU tensors for collation
- `pin_memory=True` on CUDA runs
- non-blocking transfer for both images and labels
- `channels_last` on both model and input tensors for CUDA fast paths
- AMP via `--amp-mode {auto,on,off}`
- optional compile benchmarking via `--compile-mode {auto,on,off}`

`--compile-mode auto` warms up eager and compiled paths separately, measures synchronized median train-step time, and keeps compile only when it improves throughput.

### Phase 1 Omega-Loss

The torch backend supports the Phase 1 attractor experiment through:

- `--omega-loss / --no-omega-loss`
- `--omega-lambda`
- `--idsi-lambda`
- `--omega-projector-depth`
- `--omega-hidden-dim`
- `--experiment-dir`

Policy details:
- `h` is the current 256-d representation after `FC(256) -> ReLU` and before dropout
- `T(h)` is a shallow trainable MLP projector with a final `LayerNorm`
- Phase 1.2 adds a small Layer-IDSI term when `--omega-loss` is enabled:
  `L_total = L_CE_mix + omega_lambda * L_fp + idsi_lambda * L_IDSI`
- `L_IDSI` uses matched feature spaces only: `stage1`, `stage2`, `stage3`, and `classifier_pre_head`
- Phase1.2 detaches only the Layer-IDSI denominator norm for numerical stability; no spectral normalization, memory bank, tokenization, or attention mechanism is introduced in Phase 1
- contraction behavior is an empirical hypothesis in this phase, not a guaranteed property
- validation and checkpointing keep the existing EMA, AMP, compile, early-stop, augmentation, and MixUp/CutMix policies

When `--omega-loss` is enabled, the trainer writes structured run artifacts under `--experiment-dir`:

- `config.json`
- `epoch_metrics.jsonl`
- `summary.json`
- `qualitative_notes.txt`

The metrics include total loss, CE loss, attractor loss, Layer-IDSI loss, accuracy, generalization gap, representation-variance diagnostics, global/layer IDSI distribution summaries, gradient norm, and hidden norm.

### EMA

New arguments:
- `--ema / --no-ema`
- `--ema-decay`

Policy details:
- EMA defaults to enabled with `--ema-decay 0.999`
- update order is:
  - `optimizer.step()`
  - `ema.update(model)`
- EMA tracks both parameters and buffers, including BN running statistics
- validation uses EMA weights when EMA is enabled
- best checkpoint saves also use EMA weights when EMA is enabled
- phase starts use a short EMA warmup so EMA can catch up after cosine restarts or freeze/unfreeze transitions
- mixed batches slightly lower the effective EMA decay so the shadow weights track noisier updates more quickly

### Structured Checkpoints

New checkpoints now store:

```text
{
  model: ...,
  meta: {
    checkpoint_version: 2,
    backend: ...,
    num_classes: ...,
    width_scale: ...,
    stage2_channels: ...,
    input_size: ...,
    class_names: [...],
    is_ema: true/false,
    ema_decay: ...,
    omega_enabled: true/false,
    omega_projector_depth: 1 or 2,
    omega_hidden_dim: ...
  }
}
```

Notes:
- both backends still load older plain checkpoints
- both inference backends now reconstruct model architecture from checkpoint metadata before applying weights
- when metadata is missing, both backends infer `num_classes` and stage-2 width from the saved parameter shapes
- torch inference also reconstructs the optional Phase 1 Omega branch when a checkpoint contains it, but prediction uses only classifier logits
- `--init-from` loads the live model first and then syncs EMA from that loaded model so the two states start aligned

### Multiphase LR

Arguments:
- `--phase-count`
- `--lr` with one value per phase
- `--warmup-epochs`

Phase epochs are assigned with:

```python
np.array_split(range(epochs), phase_count)
```

Rules:
- the number of `--lr` values must equal `--phase-count`
- the LR list must be monotonically non-increasing
- cosine scheduling intentionally restarts per phase
- warmup ramps from `0.1 * base_lr` to `base_lr` at the start of each phase

### Temporary Backbone Freeze

If `val_acc` does not improve for `--freeze-patience` consecutive epochs inside the current phase:
- the backbone freezes temporarily
- the classifier head trains alone for exactly `--freeze-epoch-num` epochs
- the backbone then unfreezes automatically inside the same phase
- after unfreeze, the effective LR may gain an extra cumulative downward offset from `--after-unfreeze-lr-change`, while staying above the scheduler floor and the next phase start LR
- entering the next phase always resets LR to the explicit value from `--lr`

By default, BN affine parameters freeze with the backbone.

Default timed-freeze settings:
- `--freeze-patience 8`
- `--freeze-epoch-num 10`
- `--after-unfreeze-lr-change 0.0001`

Optional advanced mode:

```text
--freeze-bn-affine false
```

That keeps BN affine parameters trainable while BN running statistics remain frozen.

## Recommended Training Command

See `best_train_commands.txt`.

## Key Training Arguments

- `--data-dir`
- `--epochs`
- `--batch-size`
- `--num-workers`
- `--phase-count`
- `--lr` (one value per phase)
- `--warmup-epochs`
- `--optimizer {adamw,sgd}`
- `--momentum`
- `--weight-decay`
- `--dropout`
- `--label-smoothing`
- `--val-split`
- `--lr-schedule {cosine,step,constant}`
- `--min-lr-ratio`
- `--step-size`
- `--gamma`
- `--grad-clip`
- `--class-weighting / --no-class-weighting`
- `--focal-loss / --no-focal-loss`
- `--focal-gamma`
- `--focal-alpha {auto,none}`
- `--mixup / --no-mixup`
- `--mixup-alpha`
- `--mixup-prob`
- `--cutmix-ratio`
- `--ema / --no-ema`
- `--ema-decay`
- `--augment / --no-augment`
- `--rotation`
- `--brightness`
- `--contrast`
- `--saturation`
- `--model-width-scale`
- `--omega-loss / --no-omega-loss`
- `--omega-lambda`
- `--idsi-lambda`
- `--omega-projector-depth`
- `--omega-hidden-dim`
- `--experiment-dir`
- `--early-stop / --no-early-stop`
- `--early-stop-metric {val_loss,val_acc}`
- `--patience`
- `--min-delta`
- `--freeze-bn-affine false`
- `--freeze-patience`
- `--freeze-epoch-num`
- `--after-unfreeze-lr-change`
- `--checkpoint`
- `--streaming / --no-streaming`
- `--amp-mode {auto,on,off}`
- `--compile-mode {auto,on,off}`
- `--init-from`
- `--num-partitions`
- `--partition`
- `--auto-next-partition / --no-auto-next-partition`
- `--partition-state`
- `--class-count` (optional override)
- `--enforce-readonly-dataset / --no-enforce-readonly-dataset`
- `--seed`
- `--device {auto,cpu,cuda}`

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider
```
