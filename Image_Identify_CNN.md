# Image_Identify_CNN Training Guide

The project uses `PyTorch` as the default backend and keeps the original `NumPy` structure intact. The active architecture in both backends is a three-stage CNN + SE classifier with a flattened dense head.

## Current Runtime

- Python version: `3.14`
- Recommended interpreter: `./.venv/Scripts/python.exe`
- Default backend: `torch`
- Legacy backend: `numpy`

## Active Architecture

Both backends now use:
- Stage 1: `Conv(3->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> SE(32) -> MaxPool`
- Stage 2: `Conv(32->round(64*scale)) -> BN -> ReLU -> Conv(round(64*scale)->round(64*scale)) -> BN -> ReLU -> SE(round(64*scale)) -> MaxPool`
- Stage 3: `Conv(round(64*scale)->128) -> BN -> ReLU -> Conv(128->128) -> BN -> ReLU -> SE(128) -> MaxPool`
- Head: `Flatten -> FC(256) -> ReLU -> Dropout(0.5) -> FC(num_classes)`

Width scaling is controlled by `--model-width-scale`.

- default `0.75` gives stage-2 width `48`
- use `--model-width-scale 1.0` for older checkpoints trained with stage-2 width `64`

`Mamba` is no longer used by the active model paths.

## Environment Setup

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset Layout

```text
Dataset/
  airplane/
  bird/
  cat/
  deer/
  dog/
  frog/
  horse/
  ship/
```

Each class should be a directory. Training auto-detects the class count from `Dataset/`; use `--class-count` only when you need to override the detected output size.

## Global Backend Switch

```text
--backend {torch,numpy}
```

## New Training Behavior

### Augmentation

When `--augment` is enabled, both backends apply the same transform order:

```text
crop -> flip -> rotation -> color jitter -> random erasing
```

Policy details:
- random resized crop remains the first geometry transform
- horizontal flip uses `p=0.5`
- `--rotation x` samples an angle in `[-x, +x]`; default `12`, recommended `<= 20` for CIFAR-like datasets
- `--brightness x`, `--contrast x`, and `--saturation x` sample factors in `[1-x, 1+x]`; defaults are `0.2`
- random erasing remains the final destructive transform with:
  - `p=0.3`
  - erased area sampled from `10%` to `20%`
  - mean-value fill using the current image channel mean

Validation rules:
- `0 <= rotation < 180`
- `0 <= brightness <= 1`
- `0 <= contrast <= 1`
- `0 <= saturation <= 1`

### MixUp and Focal Loss

New arguments:
- `--mixup / --no-mixup`
- `--mixup-alpha`
- `--mixup-prob`
- `--focal-loss / --no-focal-loss`
- `--focal-gamma`
- `--focal-alpha {auto,none}`

Policy details:
- MixUp defaults to enabled
- MixUp uses `lam ~ Beta(--mixup-alpha, --mixup-alpha)` with default `--mixup-alpha 0.2`
- MixUp is applied per batch with default probability `--mixup-prob 0.5`
- focal loss defaults to `gamma=1.5`
- `--focal-alpha auto` reuses inverse-frequency class weights as focal alpha
- if MixUp is enabled for a run, focal loss is disabled automatically for that run and the trainer falls back to cross entropy for both train and validation loss reporting

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

The post-unfreeze LR rule now works on the current effective LR, not the phase base LR, so cosine decay is preserved and the deduction remains cumulative inside the phase.

## Recommended Training Command

- See in best_train_commands.txt

## Key Training Arguments

- `--data-dir`
- `--epochs`
- `--batch-size`
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
- `--augment / --no-augment`
- `--rotation`
- `--brightness`
- `--contrast`
- `--saturation`
- `--model-width-scale`
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
.\.venv\Scripts\python.exe -m pytest tests -v
```
