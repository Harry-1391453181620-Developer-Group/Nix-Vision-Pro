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
- Stage 2: `Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> SE(64) -> MaxPool`
- Stage 3: `Conv(64->128) -> BN -> ReLU -> Conv(128->128) -> BN -> ReLU -> SE(128) -> MaxPool`
- Head: `Flatten -> FC(256) -> ReLU -> Dropout(0.5) -> FC(num_classes)`

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

When `--augment` is enabled, both backends now apply the same transform order:

```text
crop -> flip -> rotation -> color jitter -> cutout
```

Policy details:
- random resized crop remains the first geometry transform
- horizontal flip uses `p=0.5`
- rotation uses reflection padding, centered rotation, bilinear interpolation, `p=0.5`, and `+-12°`
- color jitter uses `p=0.5` with:
  - brightness: `+-0.2`
  - contrast: `+-0.2`
  - saturation: `+-0.2`
- cutout remains the final destructive transform

### Multiphase LR

New arguments:
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
- cosine scheduling now intentionally restarts per phase
- warmup ramps from `0.1 * base_lr` to `base_lr` at the start of each phase

### Temporary Backbone Freeze

If `val_acc` does not improve for `--freeze-patience` consecutive epochs inside the current phase:
- the backbone freezes temporarily
- the classifier head trains alone for exactly `--freeze-epoch-num` epochs
- the backbone then unfreezes automatically inside the same phase
- after unfreeze, the phase base LR may decrease by `--after-unfreeze-lr-change` when that deduction stays above the next phase start LR
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

## Recommended PyTorch Training Command

```powershell
python.exe train.py --backend torch --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.0004 0.0001 --warmup-epochs 3 --batch-size 64 --streaming --optimizer adamw --weight-decay 2e-4 --dropout 0.3 --label-smoothing 0.05 --balance-sampling --augment --lr-schedule cosine --min-lr-ratio 0.05 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 15 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 10 --after-unfreeze-lr-change 0.0001 --device cuda --checkpoint checkpoints/best_torch_model.pt --init-from D:\Programing_materials\Python\python_Projects\Image_Identify_CNN\checkpoints\best_torch_model.pt
```

## NumPy Training Command

```powershell
python.exe train.py --backend numpy --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.002 0.0005 --warmup-epochs 3 --batch-size 32 --streaming --optimizer adamw --weight-decay 1e-5 --dropout 0.3 --label-smoothing 0.1 --class-weighting --balance-sampling --augment --lr-schedule cosine --min-lr-ratio 0.2 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 15 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 10 --after-unfreeze-lr-change 0.0001 --checkpoint checkpoints/best_numpy_model.npz
```

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
- `--balance-sampling / --no-balance-sampling`
- `--augment / --no-augment`
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
python.exe -m pytest tests -v
```
