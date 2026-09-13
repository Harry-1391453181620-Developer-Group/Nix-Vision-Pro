# Nix Vision Pro

A Python 3.13.2 image classification project with two maintained backends:
- `backends/torch/` is the default training and inference path.
- `backends/numpy/` preserves the NumPy implementation for comparison and fallback.

Project history is versioned under `Agent_History/`. Substantive contributions are expected to update the relevant history entries along with code and docs.

The active PyTorch training architecture is Phase3.1 pure ViT:
`Image -> 8x8 patch embedding -> CLS token + Transformer Encoder -> Classifier`.

The default ViT shape is intentionally small: `patch_size=8`, `token_dim=128`, `transformer_depth=4`, `attention_heads=4`, `mlp_ratio=2.0`, and `token_pool=cls`. CNN dynamics are removed from torch training. The NumPy backend remains the legacy CNN comparison path.

PyTorch inference and GUI loading remain backward compatible with Phase1/Phase2 torch checkpoints. Checkpoint metadata or legacy weight shapes select either the new ViT model or the preserved legacy CNN+Transformer loader before weights are applied.

## Runtime Baseline

- Python: `3.13.2`
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
- Omega-loss is available in the torch trainer with `--omega-loss`; in Phase3.1 it projects the ViT CLS representation and logs CE, attractor loss, and representation-variance diagnostics.
- Layer-IDSI remains enabled by default through `--idsi-lambda 0.005` and `--token-idsi`. It monitors `patch_embedding` plus one `transformer_block_N` entry per ViT block.
- ViT controls include `--token-dim`, `--transformer-depth`, `--attention-heads`, `--transformer-mlp-ratio`, `--token-pool {mean,cls}`, `--token-positional-encoding {none,learned,sinusoidal}`, `--token-dropout`, `--transformer-layernorm {pre,post}`, `--token-omega-loss`, `--token-idsi`, and `--token-diversity-monitor`.
- Token diversity metrics operate on patch tokens only; the CLS token is excluded.
- Token metrics include token variance, inter-token variance, token norm statistics, mean pairwise cosine similarity/distance, and token collapse warnings.
- Training run artifacts are written directly under `runs/<timestamp>-.../`.
- Run metrics log `IDSI`, `IDSI mean`, `IDSI max`, and `IDSI std` from the per-sample relative fluctuation distribution, scaled by `100` for readability. Layer-wise arrays are logged under `layer_IDSI*` keys with stable `layer_IDSI_names`.
- Run metrics also log `gradient_norm` from the existing backward pass before clipping and `hidden_norm` as the mean L2 norm over monitored feature tensors.
- Train-side representation variance metrics are named `train_h_var_max`, `train_h_var_mean`, and `train_h_var_min` in JSONL files and plots.
- `train.py` can show and save metric plots with `--plot-once` after training or `--plot-real-time` during training.
- New checkpoints are structured as `model + meta`, while legacy raw checkpoints still load.
- Multiphase LR, warmup, and cosine restarts remain supported. The torch temporary backbone-freeze training policy was removed for Phase3.1.

## Training

Recommended PyTorch training command:

```powershell
$args = @(
"--backend", "torch",

"--data-dir", "Dataset",
"--device", "cuda",
"--num-workers", "4",
"--streaming",
"--seed", "42",
"--class-count", "13",

"--amp-mode", "auto",
"--compile-mode", "auto",

"--epochs", "120",
"--phase-count", "1", 
"--batch-size", "128",

"--optimizer", "adamw",
"--lr", "0.0003",
"--lr-schedule", "cosine",
"--min-lr-ratio", "0.2",
"--warmup-epochs", "5",
"--weight-decay", "1e-5",
"--grad-clip", "5.0",

"--dropout", "0.10",
"--label-smoothing", "0.04",
"--no-focal-loss",

"--augment",
"--rotation", "20",
"--brightness", "0.25",
"--contrast", "0.25",
"--saturation", "0.25",

"--mixup",
"--mixup-alpha", "0.3",
"--mixup-prob", "0",
"--cutmix-ratio", "0.2",

"--ema",
"--ema-decay", "0.9993",

"--early-stop",
"--early-stop-metric", "val_acc",
"--patience", "50",
"--min-delta", "0.000001",

"--checkpoint", "checkpoints/best_torch_model.pt",

"--omega-loss",
"--omega-lambda", "0.05",
"--omega-projector-depth", "1",
"--omega-hidden-dim", "256",

"--idsi-lambda", "0.005",

"--tokenize",
"--token-dim", "128",
"--transformer-depth", "4",
"--attention-heads", "4",
"--transformer-mlp-ratio", "2.0",
"--token-pool", "cls",
"--token-positional-encoding", "learned",
"--token-dropout", "0.10",
"--transformer-layernorm", "pre",
"--token-omega-loss",
"--token-idsi",
"--token-diversity-monitor",

"--json-dir", "runs",
"--plot-real-time",
"--plot-output-format", "png",

"--enforce-readonly-dataset",

"--num-partitions", "1",
"--partition", "0"

)

python.exe train.py @args
```

Legacy NumPy training remains available:

```powershell
python.exe train.py --backend numpy --data-dir Dataset --epochs 100 --phase-count 2 --lr 0.002 0.0005 --warmup-epochs 3 --batch-size 32 --streaming --optimizer adamw --weight-decay 1e-5 --dropout 0.3 --label-smoothing 0.05 --class-weighting --focal-loss --focal-gamma 1.5 --focal-alpha auto --augment --rotation 12 --brightness 0.2 --contrast 0.2 --saturation 0.2 --model-width-scale 0.75 --lr-schedule cosine --min-lr-ratio 0.2 --grad-clip 5.0 --early-stop --early-stop-metric val_acc --patience 15 --min-delta 0.001 --freeze-bn-affine false --freeze-patience 8 --freeze-epoch-num 10 --after-unfreeze-lr-change 0.0001 --mixup --mixup-alpha 0.2 --mixup-prob 0.4 --cutmix-ratio 0.5 --ema --ema-decay 0.999 --checkpoint checkpoints/best_numpy_model.npz
```

## Experiment Plotting

Runs write per-epoch metrics to `runs/<timestamp>-.../epoch_metrics.jsonl`.
To show a plot after training finishes and save an image beside the JSONL file:

```powershell
.\.venv\Scripts\python.exe train.py --backend torch --omega-loss --omega-lambda 0.05 --idsi-lambda 0.005 --plot-once
```

To open the plot window at training start, refresh it after each epoch, and save the final image:

```powershell
.\.venv\Scripts\python.exe train.py --backend torch --omega-loss --omega-lambda 0.05 --idsi-lambda 0.005 --plot-real-time
```

Plot-related options are `--json-dir`, `--plot-output-format {png,jpg,jpeg}`, and `--plot-output-dir`.
The plotter keeps all metrics in one matplotlib window, includes global and layer-wise IDSI panels, and adapts dynamically to however many monitored layers are present in the JSONL rows.
Token metrics are optional scalar panels, so older Phase 1/1.2/Phase2 JSONL files remain readable.
`plot.py` is a helper module used by `train.py`; it is not a standalone command.

## Checkpoints

- New checkpoints save a structured payload with the model weights under `model` plus metadata under `meta`.
- The metadata includes `checkpoint_version`, `backend`, `architecture`, `num_classes`, `input_size`, `class_names`, and whether the saved weights are EMA weights.
- Phase3.1 torch checkpoints also save `patch_size`, `vit_depth`, `token_dim`, `attention_heads`, `pool_type`, transformer settings, token pooling, and positional encoding.
- Torch checkpoints may also include Omega metadata: `omega_enabled`, `omega_projector_depth`, and `omega_hidden_dim`; inference reconstructs this branch when present but still predicts from logits only.
- Phase1/Phase2 torch checkpoints remain loadable through the preserved legacy CNN+Transformer model path.
- `load_weights()` still accepts older plain state-dict checkpoints, so older training runs remain usable.
- Torch and NumPy inference now rebuild the model from checkpoint metadata first and fall back to legacy weight-shape inference when metadata is missing.
- If you pass `--class-count` while loading weights, conflicting overrides fail clearly. `--model-width-scale` conflicts are enforced for legacy CNN torch checkpoints and NumPy checkpoints.
- Load only trusted checkpoints. The PyTorch loader prefers `weights_only=True`, but older interpreter combinations can still fall back to normal `torch.load()` for compatibility.

## Inference

```powershell
python.exe predict.py --backend torch Dataset/airplane/0000001.jpg --weights checkpoints/best_torch_model.pt --probabilities --top-k 3 --device cuda
python.exe predict.py --backend numpy Dataset/airplane/0000001.jpg --weights checkpoints/best_numpy_model.npz --probabilities --top-k 3
```

## GUI

```powershell
python.exe gui.py --backend torch --data-dir Dataset
python.exe gui.py --backend numpy --data-dir Dataset
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider
```
