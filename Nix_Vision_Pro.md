# Nix Vision Pro Training Guide

The project uses `PyTorch` as the default backend and keeps the original `NumPy` structure intact. The active torch training architecture is Phase3.1 pure ViT; the NumPy backend remains the legacy CNN comparison path.

## Current Runtime

- Python version: `3.13.2`
- Recommended interpreter: `./.venv/Scripts/python.exe`
- Default backend: `torch`
- Legacy backend: `numpy`
- Active dataset source: real images stored under `Dataset/` by default
- `Agent_History/` is versioned; substantive contributions should update the relevant history entries together with code and docs

## Active Architecture

Torch training now uses:
- Patch embedding: image patches of `8x8` pixels projected to token dimension `128`
- Transformer encoder: `4` residual transformer blocks, `4` attention heads, MLP ratio `2.0`
- Pooling: CLS token by default
- Classifier: dropout `0.10` then a linear class head

The active torch path is:

```text
Image -> Patch Embedding -> CLS + Transformer Encoder -> Classifier
```

CNN stages are not part of Phase3.1 torch training. The old torch CNN+Transformer implementation is retained only as `TorchLegacyCNN` so Phase1/Phase2 checkpoints can still be inspected and loaded by inference and GUI tools.

The NumPy backend still uses the previous three-stage CNN + SE architecture and keeps its existing width-scale and freeze controls.

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

This switch is supported by `train.py`, `predict.py`, and `gui.py`.

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

### Phase3.1 Omega-Loss and Layer-IDSI

The torch backend supports the Phase 1 attractor experiment through:

- `--omega-loss / --no-omega-loss`
- `--omega-lambda`
- `--idsi-lambda`
- `--omega-projector-depth`
- `--omega-hidden-dim`
- `--experiment-dir`

Policy details:
- `h` is the current ViT CLS representation, or the mean-pooled patch representation when `--token-pool mean` is selected
- `T(h)` is a shallow trainable MLP projector with a final `LayerNorm`
- Phase3.1 adds a small Layer-IDSI term when `--token-idsi` is enabled:
  `L_total = L_CE_mix + omega_lambda * L_fp + idsi_lambda * L_IDSI`
- `L_IDSI` uses matched token spaces only: `patch_embedding`, then `transformer_block_1` through `transformer_block_N`
- transformer block IDSI is measured on each block output after the final residual connection
- contraction behavior is an empirical hypothesis in this phase, not a guaranteed property
- validation and checkpointing keep the existing EMA, AMP, compile, early-stop, augmentation, and MixUp/CutMix policies

When `--omega-loss` is enabled, the trainer writes structured run artifacts under `--experiment-dir`:

- `config.json`
- `epoch_metrics.jsonl`
- `summary.json`
- `qualitative_notes.txt`

The metrics include total loss, CE loss, attractor loss, Layer-IDSI loss, accuracy, generalization gap, representation-variance diagnostics, global/layer IDSI distribution summaries, gradient norm, and hidden norm.

### Phase3.1 ViT Token Dynamics

The torch backend uses patch-token dynamics by default through:

- `--tokenize / --no-tokenize`
- `--token-dim`
- `--transformer-depth`
- `--attention-heads`
- `--transformer-mlp-ratio`
- `--token-pool {mean,cls}`
- `--token-positional-encoding {none,learned,sinusoidal}`
- `--token-dropout`
- `--transformer-layernorm {pre,post}`
- `--token-omega-loss / --no-token-omega-loss`
- `--token-idsi / --no-token-idsi`
- `--token-diversity-monitor / --no-token-diversity-monitor`

Policy details:
- tokenization is on by default and `--no-tokenize` is rejected for torch training because Phase3.1 is pure ViT
- the default token path uses `token_dim=128`, `transformer_depth=4`, `attention_heads=4`, `transformer_mlp_ratio=2.0`, and `token_pool=cls`
- patch size is fixed at `8` for the initial Phase3.1 baseline
- token Omega uses the ViT representation and Omega projector
- token Layer-IDSI monitors `patch_embedding` and one `transformer_block_N` row per configured transformer block
- token Layer-IDSI does not monitor attention heads, attention projections, FFN sublayers, or LayerNorm submodules separately
- token diversity metrics track patch-token variance, inter-token variance, token norm statistics, pairwise cosine similarity/distance, and collapse warnings; the CLS token is excluded from these diversity calculations

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
- phase starts use a short EMA warmup so EMA can catch up after cosine restarts
- mixed batches slightly lower the effective EMA decay so the shadow weights track noisier updates more quickly

### Structured Checkpoints

New torch ViT checkpoints now store:

```text
{
  model: ...,
  meta: {
    checkpoint_version: 3,
    backend: "torch",
    architecture: "vit",
    num_classes: ...,
    input_size: ...,
    class_names: [...],
    is_ema: true/false,
    ema_decay: ...,
    omega_enabled: true/false,
    omega_projector_depth: 1 or 2,
    omega_hidden_dim: ...,
    patch_size: 8,
    vit_depth: ...,
    token_dim: ...,
    attention_heads: ...,
    pool_type: "cls" or "mean"
  }
}
```

Notes:
- both backends still load older plain checkpoints
- both inference backends now reconstruct model architecture from checkpoint metadata before applying weights
- when metadata is missing, torch infers whether the checkpoint is ViT, legacy CNN, or legacy CNN+Transformer from saved parameter names
- torch inference also reconstructs the optional Omega branch when a checkpoint contains it, but prediction uses only classifier logits
- Phase1/Phase2 torch checkpoints use the preserved legacy model path and should load without crashing
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

### Torch Freeze Policy Removed

Torch Phase3.1 training does not expose the former temporary backbone-freeze policy. These torch arguments were removed:

- `--model-width-scale`
- `--freeze-bn-affine`
- `--freeze-patience`
- `--freeze-epoch-num`
- `--after-unfreeze-lr-change`

The NumPy backend remains legacy CNN code and may still expose its previous width-scale and freeze controls.

## Recommended Training Command

See `best_train_commands.txt`.

## Key Training Arguments

- `--help-md`
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
- `--omega-loss / --no-omega-loss`
- `--omega-lambda`
- `--idsi-lambda`
- `--omega-projector-depth`
- `--omega-hidden-dim`
- `--tokenize / --no-tokenize` (`--no-tokenize` is rejected by torch Phase3.1)
- `--token-dim`
- `--transformer-depth`
- `--attention-heads`
- `--transformer-mlp-ratio`
- `--token-pool {mean,cls}`
- `--token-positional-encoding {none,learned,sinusoidal}`
- `--token-dropout`
- `--transformer-layernorm {pre,post}`
- `--token-omega-loss / --no-token-omega-loss`
- `--token-idsi / --no-token-idsi`
- `--token-diversity-monitor / --no-token-diversity-monitor`
- `--allow-unlabeled-root / --no-allow-unlabeled-root`
- `--experiment-dir`
- `--json-dir`
- `--plot-once`
- `--plot-real-time`
- `--plot-output-format {png,jpg,jpeg}`
- `--plot-output-dir`
- `--early-stop / --no-early-stop`
- `--early-stop-metric {val_loss,val_acc}`
- `--patience`
- `--min-delta`
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

## Inference And GUI

Inference examples:

```powershell
python.exe predict.py --backend torch Dataset/airplane/0000001.jpg --weights checkpoints/best_torch_model.pt --probabilities --top-k 3 --device cuda
python.exe predict.py --backend numpy Dataset/airplane/0000001.jpg --weights checkpoints/best_numpy_model.npz --probabilities --top-k 3
```

GUI examples:

```powershell
python.exe gui.py --backend torch --data-dir Dataset
python.exe gui.py --backend numpy --data-dir Dataset
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider
```
