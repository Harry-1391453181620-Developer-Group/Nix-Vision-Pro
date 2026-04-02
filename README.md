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

- `--augment` now applies a shared RandAugment-style policy in `utils/training.py` for both backends.
- Each training image samples two ops from the shared pool: `rotate`, `brightness`, `contrast`, `saturation`, `sharpness`, `posterize`, `solarize`, `autocontrast`, `equalize`, `invert`, and `cutout`.
- `--rotation`, `--brightness`, `--contrast`, and `--saturation` still control the user-facing strength limits.
- Batch mixing is now a single gate: `--mixup-prob 0.4` decides whether to mix, then `--cutmix-ratio 0.5` decides between CutMix and MixUp for that batch.
- Validation is always clean: no RandAugment, no MixUp, and no CutMix.
- Mixed batches force label smoothing to `0` for that batch and only use cross entropy. Non-mixed batches can still use focal loss.
- `--ema` is enabled by default with `--ema-decay 0.999`.
- EMA updates run immediately after `optimizer.step()`, evaluate the EMA weights for validation, and save best checkpoints from EMA weights.
- EMA tracks the full model state, including BN buffers, so validation does not drift from stale running statistics.
- New checkpoints are structured as `model + meta`, while legacy raw checkpoints still load.
- Multiphase LR, warmup, cosine restarts, and temporary backbone freeze remain supported in both training backends.

## Training

Recommended PyTorch training command:

```powershell
python.exe train.py --backend torch --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.0004 0.0002 --warmup-epochs 3 --batch-size 64 --streaming --optimizer adamw --seed 42 --weight-decay 2e-4 --dropout 0.3 --label-smoothing 0.05 --augment --no-class-weighting --lr-schedule cosine --min-lr-ratio 0.08 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 20 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 6 --after-unfreeze-lr-change 0.00004 --focal-loss --focal-gamma 1.8 --focal-alpha auto --rotation 12 --brightness 0.2 --contrast 0.2 --saturation 0.2 --model-width-scale 1.5 --mixup --mixup-alpha 0.1 --mixup-prob 0.4 --cutmix-ratio 0.5 --ema --ema-decay 0.999 --device cuda --class-count 62 --checkpoint checkpoints/best_torch_model.pt --init-from checkpoints/best_torch_model.pt
```

Legacy NumPy training remains available:

```powershell
python.exe train.py --backend numpy --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.002 0.0005 --warmup-epochs 3 --batch-size 32 --streaming --optimizer adamw --weight-decay 1e-5 --dropout 0.3 --label-smoothing 0.05 --class-weighting --focal-loss --focal-gamma 1.5 --focal-alpha auto --augment --rotation 12 --brightness 0.2 --contrast 0.2 --saturation 0.2 --model-width-scale 0.75 --lr-schedule cosine --min-lr-ratio 0.2 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 15 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 10 --after-unfreeze-lr-change 0.0001 --mixup --mixup-alpha 0.2 --mixup-prob 0.4 --cutmix-ratio 0.5 --ema --ema-decay 0.999 --checkpoint checkpoints/best_numpy_model.npz
```

## Checkpoints

- New checkpoints save a structured payload with the model weights under `model` plus metadata under `meta`.
- The metadata includes the checkpoint version, backend, and whether the saved weights are EMA weights.
- `load_weights()` still accepts older plain state-dict checkpoints, so older training runs remain usable.
- Load only trusted checkpoints. The PyTorch loader prefers `weights_only=True`, but older interpreter combinations can still fall back to normal `torch.load()` for compatibility.

## Inference

```powershell
python.exe predict.py --backend torch Dataset/airplane/0000001.jpg --weights checkpoints/best_torch_model.pt --probabilities --top-k 3 --device cuda --model-width-scale 0.75
python.exe predict.py --backend numpy Dataset/airplane/0000001.jpg --weights checkpoints/best_numpy_model.npz --probabilities --top-k 3 --model-width-scale 0.75
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider
```
