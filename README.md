# Image_Identify_CNN

A Python 3.14 image classification project with two maintained backends:
- `backends/torch/` is the default training and inference path.
- `backends/numpy/` preserves the NumPy implementation for comparison and fallback.

The active architecture in both backends is width-scaled in stage 2:
`[Conv(32), Conv(32), SE, Pool] -> [Conv(round(64*scale)), Conv(round(64*scale)), SE, Pool] -> [Conv(128), Conv(128), SE, Pool] -> Flatten -> FC(256) -> Dropout(0.5) -> FC(num_classes)`.

Default width scale is `0.75`, so the active stage-2 width is `48`. Use `--model-width-scale 1.0` for older `64`-channel checkpoints.

## Runtime Baseline

- Python: `3.14`
- Virtual environment: `.venv`
- Default backend: `torch`
- Legacy backend: `numpy`

## Install

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The current dependency set covers the whole project:
- `torch`
- `numpy`
- `pillow`
- `opencv-python`
- `pytest`

## Main Entry Points

- `train.py` - backend-dispatching training entrypoint
- `predict.py` - backend-dispatching inference entrypoint
- `gui.py` - backend-dispatching GUI entrypoint
- `model.py` - compatibility facade exposing the default model class

## Training Policy Highlights

- Augmentation order is now `crop -> flip -> rotation -> color jitter -> random erasing`.
- `--rotation`, `--brightness`, `--contrast`, and `--saturation` control augmentation strength directly.
- Random erasing uses `p=0.3`, erases `10%-20%` of image area, and fills with the per-image channel mean.
- `--mixup` and `--mixup-prob` enable Beta(0.2, 0.2) MixUp.
- `--focal-loss`, `--focal-gamma`, and `--focal-alpha` replace the old balance-sampling path.
- If MixUp is enabled for a run, focal loss is automatically disabled for that run and the trainer falls back to cross entropy.
- Multiphase LR uses `--phase-count` and a list passed to `--lr`.
- Phase boundaries are computed with `np.array_split(range(epochs), phase_count)`.
- Cosine schedule intentionally restarts per phase.
- Optional warmup uses `--warmup-epochs` and ramps from `0.1 * base_lr` to `base_lr` at each phase start.
- If `val_acc` stalls inside a phase, the backbone freezes temporarily, then unfreezes inside the same phase.
- `--freeze-bn-affine false` keeps BN affine parameters trainable while BN running stats stay frozen.
- `--after-unfreeze-lr-change` builds a cumulative within-phase LR offset from the current effective LR, clamped to the scheduler floor and capped so it never undercuts the next phase start LR.

## Training

Recommended PyTorch training command:

```powershell
python.exe train.py --backend torch --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.0004 0.00015 --warmup-epochs 3 --batch-size 128 --streaming --optimizer adamw --weight-decay 4e-4 --dropout 0.35 --label-smoothing 0.1 --class-weighting --focal-loss --focal-gamma 1.5 --focal-alpha auto --augment --rotation 12 --brightness 0.2 --contrast 0.2 --saturation 0.2 --model-width-scale 0.75 --lr-schedule cosine --min-lr-ratio 0.02 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 20 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 5 --freeze-epoch-num 6 --after-unfreeze-lr-change 0.00008 --checkpoint checkpoints/best_torch_model.pt --device cuda
```

Legacy NumPy training remains available:

```powershell
python.exe train.py --backend numpy --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.002 0.0005 --warmup-epochs 3 --batch-size 32 --streaming --optimizer adamw --weight-decay 1e-5 --dropout 0.3 --label-smoothing 0.1 --class-weighting --focal-loss --focal-gamma 1.5 --focal-alpha auto --augment --rotation 12 --brightness 0.2 --contrast 0.2 --saturation 0.2 --model-width-scale 0.75 --lr-schedule cosine --min-lr-ratio 0.2 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 15 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 10 --after-unfreeze-lr-change 0.0001 --checkpoint checkpoints/best_numpy_model.npz
```

## Inference

```powershell
python.exe predict.py --backend torch Dataset/airplane/0000001.jpg --weights checkpoints/best_torch_model.pt --probabilities --top-k 3 --device cuda --model-width-scale 0.75
python.exe predict.py --backend numpy Dataset/airplane/0000001.jpg --weights checkpoints/best_numpy_model.npz --probabilities --top-k 3 --model-width-scale 0.75
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```
