# Image_Identify_CNN Training Guide

The project now uses `PyTorch` as the default backend and keeps the original `NumPy` structure intact. The active architecture in both backends is now a three-stage CNN + SE classifier with a flattened dense head.

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

Each class should be a directory. Training now auto-detects the class count from `Dataset/`; use `--class-count` only when you need to override the detected output size.

## Global Backend Switch

```text
--backend {torch,numpy}
```

## Recommended PyTorch Training Command

```powershell
python.exe train.py --backend torch --data-dir Dataset --epochs 100 --batch-size 64 --streaming --optimizer adamw --lr 3e-3 --weight-decay 1e-4 --dropout 0.3 --label-smoothing 0.05 --class-weighting --balance-sampling --augment --lr-schedule cosine --min-lr-ratio 0.05 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 50 --min-delta 0.002 --device cuda --checkpoint checkpoints/best_torch_model.pt
```

## NumPy Training Command

```powershell
python.exe train.py --backend numpy --data-dir Dataset --epochs 100 --batch-size 64 --streaming --optimizer adamw --lr 3e-3 --weight-decay 1e-4 --dropout 0.3 --label-smoothing 0.05 --class-weighting --balance-sampling --augment --lr-schedule cosine --min-lr-ratio 0.05 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 50 --min-delta 0.002 --checkpoint checkpoints/best_numpy_model.npz
```

## Key Training Arguments

- `--data-dir`
- `--epochs`
- `--batch-size`
- `--lr`
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




