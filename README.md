# Image_Identify_CNN

A Python 3.14 image classification project with two maintained backends:
- `backends/torch/` is the default training and inference path.
- `backends/numpy/` preserves the NumPy implementation for comparison and fallback.

The active architecture in both backends is width-scaled in stage 2:
`[Conv(32), Conv(32), SE, Pool] -> [Conv(round(64*scale)), Conv(round(64*scale)), SE, Pool] -> [Conv(128), Conv(128), SE, Pool] -> Flatten -> FC(256) -> Dropout(0.5) -> FC(num_classes)`.

Default width scale is `0.75`, so the default stage-2 width is `48`. When you load a checkpoint, inference now reconstructs `num_classes`, `width_scale`, `stage2_channels`, and `input_size` from checkpoint metadata or legacy weight shapes, so `--model-width-scale` is usually unnecessary for saved checkpoints.

## Runtime Baseline

- Python: `3.14`
- Virtual environment: `.venv`
- Default backend: `torch`
- Legacy backend: `numpy`
- Active training data source: real image files under `Dataset/` or an explicit `--data-dir`

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
- `matplotlib`
- `pytest`

## Main Entry Points

- `train.py` - backend-dispatching training entrypoint
- `predict.py` - backend-dispatching inference entrypoint
- `gui.py` - backend-dispatching GUI entrypoint
- `model.py` - compatibility facade exposing the default model class

## Dataset Contract

Training and inference read labeled image files from `Dataset/<class_name>/...` by default.
The runtime class list is detected from the active dataset layout, and checkpoint-backed inference prefers checkpoint metadata when it matches the saved model.
This repository no longer includes built-in synthetic or Wikimedia dataset builders because they were stale with the active real-image runtime path.

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
- The torch trainer now uses `DataLoader`, worker seeding, contiguous tensor collation, `pin_memory` on CUDA, and non-blocking transfer for both inputs and labels.
- Torch MixUp and CutMix now run on device tensors after transfer instead of round-tripping through NumPy on the CPU path.
- Torch training enables `channels_last`, `torch.backends.cudnn.benchmark`, and AMP automatically on supported CUDA runs.
- `--compile-mode auto` benchmarks eager vs compiled train steps after warmup and keeps `torch.compile` only when it actually improves median step time.
- Phase 1 Omega-loss is available in the torch trainer with `--omega-loss`; it adds a shallow projector on the existing 256-d penultimate representation and logs CE, attractor loss, and representation-variance diagnostics.
- Phase 1.2 Layer-IDSI is enabled for Omega runs by default with `--idsi-lambda 0.005`. It adds a numerically small layer-wise stability loss on matched feature spaces for `stage1`, `stage2`, `stage3`, and `classifier_pre_head`.
- Training run artifacts are written directly under `runs/<timestamp>-.../`.
- Run metrics log `IDSI`, `IDSI mean`, `IDSI max`, and `IDSI std` from the per-sample relative fluctuation distribution, scaled by `100` for readability. Layer-wise arrays are logged under `layer_IDSI*` keys with stable `layer_IDSI_names`.
- Run metrics also log `gradient_norm` from the existing backward pass before clipping and `hidden_norm` as the mean L2 norm over monitored feature tensors.
- Train-side representation variance metrics are named `train_h_var_max`, `train_h_var_mean`, and `train_h_var_min` in JSONL files and plots.
- `train.py` can show and save metric plots with `--plot-once` after training or `--plot-real-time` during training.
- New checkpoints are structured as `model + meta`, while legacy raw checkpoints still load.
- Multiphase LR, warmup, cosine restarts, and temporary backbone freeze remain supported in both training backends.

## Training

Recommended PyTorch training command:

```powershell
python.exe train.py --backend torch --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.0004 0.0002 --warmup-epochs 3 --batch-size 64 --streaming --num-workers 4 --optimizer adamw --seed 42 --weight-decay 2e-4 --dropout 0.3 --label-smoothing 0.05 --augment --no-class-weighting --lr-schedule cosine --min-lr-ratio 0.08 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 20 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 6 --after-unfreeze-lr-change 0.00004 --focal-loss --focal-gamma 1.8 --focal-alpha auto --rotation 12 --brightness 0.2 --contrast 0.2 --saturation 0.2 --model-width-scale 1.5 --mixup --mixup-alpha 0.1 --mixup-prob 0.4 --cutmix-ratio 0.5 --ema --ema-decay 0.999 --amp-mode auto --compile-mode auto --device cuda --checkpoint checkpoints/best_torch_model.pt --init-from checkpoints/best_torch_model.pt
```

Legacy NumPy training remains available:

```powershell
python.exe train.py --backend numpy --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.002 0.0005 --warmup-epochs 3 --batch-size 32 --streaming --optimizer adamw --weight-decay 1e-5 --dropout 0.3 --label-smoothing 0.05 --class-weighting --focal-loss --focal-gamma 1.5 --focal-alpha auto --augment --rotation 12 --brightness 0.2 --contrast 0.2 --saturation 0.2 --model-width-scale 0.75 --lr-schedule cosine --min-lr-ratio 0.2 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 15 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 10 --after-unfreeze-lr-change 0.0001 --mixup --mixup-alpha 0.2 --mixup-prob 0.4 --cutmix-ratio 0.5 --ema --ema-decay 0.999 --checkpoint checkpoints/best_numpy_model.npz
```

## Experiment Plotting

Runs write per-epoch metrics to `runs/<timestamp>-.../epoch_metrics.jsonl`.
To show a plot after training finishes and save an image beside the JSONL file:

```powershell
.\.venv312\Scripts\python.exe train.py --backend torch --omega-loss --omega-lambda 0.05 --idsi-lambda 0.005 --plot-once
```

To open the plot window at training start, refresh it after each epoch, and save the final image:

```powershell
.\.venv312\Scripts\python.exe train.py --backend torch --omega-loss --omega-lambda 0.05 --idsi-lambda 0.005 --plot-real-time
```

Plot-related options are `--json-dir`, `--plot-output-format {png,jpg,jpeg}`, and `--plot-output-dir`.
The plotter keeps all metrics in one matplotlib window, includes global and layer-wise IDSI panels, and adapts dynamically to however many monitored layers are present in the JSONL rows.
`plot.py` is a helper module used by `train.py`; it is not a standalone command.

## Checkpoints

- New checkpoints save a structured payload with the model weights under `model` plus metadata under `meta`.
- The metadata includes `checkpoint_version`, `backend`, `num_classes`, `width_scale`, `stage2_channels`, `input_size`, `class_names`, and whether the saved weights are EMA weights.
- Torch checkpoints may also include Phase 1 Omega metadata: `omega_enabled`, `omega_projector_depth`, and `omega_hidden_dim`; inference reconstructs this branch when present but still predicts from logits only.
- `load_weights()` still accepts older plain state-dict checkpoints, so older training runs remain usable.
- Torch and NumPy inference now rebuild the model from checkpoint metadata first and fall back to legacy weight-shape inference when metadata is missing.
- If you pass `--class-count` or `--model-width-scale` while also loading weights, conflicting overrides now fail clearly instead of silently building the wrong model shape.
- Load only trusted checkpoints. The PyTorch loader prefers `weights_only=True`, but older interpreter combinations can still fall back to normal `torch.load()` for compatibility.

## Inference

```powershell
python.exe predict.py --backend torch Dataset/airplane/0000001.jpg --weights checkpoints/best_torch_model.pt --probabilities --top-k 3 --device cuda
python.exe predict.py --backend numpy Dataset/airplane/0000001.jpg --weights checkpoints/best_numpy_model.npz --probabilities --top-k 3
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider
```
