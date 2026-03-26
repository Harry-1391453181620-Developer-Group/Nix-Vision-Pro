# Image_Identify_CNN

A Python 3.14 image classification project with two maintained backends:
- `backends/torch/` is the default training and inference path.
- `backends/numpy/` preserves the NumPy implementation for comparison and fallback.

The active architecture in both backends is:
`[Conv(32), Conv(32), SE, Pool] -> [Conv(64), Conv(64), SE, Pool] -> [Conv(128), Conv(128), SE, Pool] -> Flatten -> FC(256) -> Dropout(0.5) -> FC(num_classes)`.

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

- Augmentation order is now `crop -> flip -> rotation -> color jitter -> cutout`.
- Rotation uses reflection padding, centered rotation, bilinear interpolation, `p=0.5`, and `+-12°`.
- Color jitter uses `p=0.5` with brightness / contrast / saturation ranges of `+-0.2`.
- Multiphase LR uses `--phase-count` and a list passed to `--lr`.
- Phase boundaries are computed with `np.array_split(range(epochs), phase_count)`.
- Cosine schedule now restarts per phase intentionally.
- Optional warmup uses `--warmup-epochs` and ramps from `0.1 * base_lr` to `base_lr` at each phase start.
- If `val_acc` stalls for 8 epochs inside a phase, the backbone freezes temporarily for 10 epochs, then unfreezes inside the same phase.
- `--freeze-bn-affine false` keeps BN affine parameters trainable while BN running stats stay frozen.
- `--after-unfreeze-lr-change` applies an additive LR drop after unfreeze only when the resulting LR stays positive and does not go below the next phase start LR.

## Training

Recommended PyTorch training command:

```powershell
python.exe train.py --backend torch --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.002 0.0005 --warmup-epochs 3 --batch-size 32 --streaming --optimizer adamw --weight-decay 1e-5 --dropout 0.5 --label-smoothing 0.1 --class-weighting --balance-sampling --augment --lr-schedule cosine --min-lr-ratio 0.2 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 5 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 10 --after-unfreeze-lr-change 0.0001 --checkpoint checkpoints/best_torch_model.pt --device cuda
```

Legacy NumPy training remains available:

```powershell
python.exe train.py --backend numpy --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.002 0.0005 --warmup-epochs 3 --batch-size 32 --streaming --optimizer adamw --weight-decay 1e-5 --dropout 0.5 --label-smoothing 0.1 --class-weighting --balance-sampling --augment --lr-schedule cosine --min-lr-ratio 0.2 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 5 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 10 --after-unfreeze-lr-change 0.0001 --checkpoint checkpoints/best_numpy_model.npz
```

## Inference

```powershell
python.exe predict.py --backend torch Dataset\airplane\0000001.jpg --weights checkpoints\best_torch_model.pt --probabilities --top-k 3 --device cuda
python.exe predict.py --backend numpy Dataset\airplane\0000001.jpg --weights checkpoints\best_numpy_model.npz --probabilities --top-k 3
```

## Tests

```powershell
python.exe -m pytest tests -v
```
