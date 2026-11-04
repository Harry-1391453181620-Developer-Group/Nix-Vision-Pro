# Nix Vision Pro — MAOL and CE Control

CNN-only image classification implementing **MAOL (Multi-Attractor Optimized
Learning)**.  The active theory is `MAOL.docx`; the old MAOIDL Omega/IDSI system,
the token transformer, and the NumPy backend were removed.

## What MAOL is

Each class is a leaf **well**: a center `c` and radius `r` in the 256-d embedding
space.  Wells are nested in a tree defined by the dataset directory hierarchy.
Only leaf centers/radii are learnable; parent centers/radii are derived bottom-up
from child centers and subtree training counts.

Classification in MAOL mode uses a prototype head, not a linear classifier:

```text
logits_c = -||h - c_c||^2 / tau
```

`h = L2Normalize(ReLU(fc1(flatten(features))))`; the prototype head and all five
geometry losses share this unit-sphere representation.  No additional module is
inserted into the backbone.

`--maol-mode none` is the **CE linear control**: it uses the same CNN backbone,
raw `ReLU(fc1(...))`, a plain `Linear(256, num_classes)` head, and CE only.  It
does not build wells, radii, `c_EMA`, or `tau`, and it skips MAOL warm-up/probe.

The six losses are:

```text
L_MAOL = L_CE
       + lambda_1 * L_intra
       + lambda_2 * L_sep
       + lambda_3 * L_nest
       + lambda_4 * L_cov
       + lambda_5 * L_center
```

All terms use the effective-count denominators from the updated document:
`L_CE`/`L_intra` over batch size, `L_sep` over valid sibling pairs, `L_cov`/
`L_center` over present leaf classes, `L_nest` over non-root nodes.

During warm-up, leaf centers are recomputed at the start of each epoch from the
current backbone and kept frozen (`requires_grad=False`); only `L_CE` is used.
For the first 100 warm-up iterations `tau=1.0`; afterwards
`tau_warm = mean ||h_i - c_yi||^2`.  At the end of warm-up the document's
statistics (`c_EMA`, `q`, `d_intra`, `d_inter`, confusion weights, margins,
`tau0 = d_intra^2`) are initialized and the wells are unfrozen.

## Dataset layout

The tree is encoded by directories.  A leaf is a directory containing images and
no child class directories:

```text
Dataset/
  living-matters/
    animals/
      cat/
      dog/
      ...
    plants/
      daisy/
      ...
  non-living-matters/
    artificial/
      airplane/
      ...
    natural/
      mountain/
      ...
```

Leaf directory names are the class labels and must be unique.  A directory may
not contain both images and nested class directories.

The current dataset has 18 leaf classes and 34,233 images.

## Architecture

- Five stages: `Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> SE -> MaxPool`
- Width scale over `(32, 64, 128, 256, 256)`; default `1.5`
- `h = L2Normalize(ReLU(fc1(flatten(features))))`, unit norm; the classifier and all geometry losses use this same normalized `h`
- `PrototypeHead` with 18 leaf wells (current dataset)
- Input `224 x 224`, total stride `/32`

## Install

```powershell
python -m venv --system-site-packages venv313
.\venv313\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On the DGX Spark (Linux), create a Linux venv instead; the Windows `venv313`
binaries are not usable there:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.13 is the tested runtime.  Training was validated on the DGX Spark via
Syncthing and locally on CPU for smoke tests.

## Training workflow

### 1. Warm-up only

Warm-up must be at least 5 epochs; the default is 15 because MAOL warm-up does
**not** use a pretrained `--init-from` backbone.  It uses clean images by
default (RandAugment is off), recomputes leaf centers at every epoch, and freezes
them.  `tau` is fixed at `1.0` for the first 100 iterations; afterwards the
measured `tau_warm` is used.  If `tau_warm < --maol-tau-floor` (default `1e-2`),
the code falls back to `tau=1.0` instead of using a collapsed value.

Warm-up LR defaults to `2 * lr[0]`; override with `--maol-warmup-lr`.

```powershell
python train.py `
  --data-dir Dataset `
  --device cuda `
  --batch-size 128 `
  --num-workers 8 `
  --model-width-scale 1.5 `
  --maol-warmup-epochs 15 `
  --maol-warmup-only `
  --checkpoint checkpoints/maol_model.pt `
  --maol-warmup-ckpt checkpoints/maol_warmup.pt `
  --json-dir runs `
  --no-early-stop
```

At the end, the sanity gate checks:

```text
val_acc >= 2 / num_classes
d_inter / d_intra >= 1.3
q_max / q_median <= 10
tau_floor <= tau0 <= 10
```

If any check fails, the run writes `warmup_diagnostics.json` and
`warmup_rejected.json`, prints the failures, and exits with code `2`.  No valid
`maol_warmup.pt` is saved; fix the epoch count, LR, or feature scale and rerun.
Use `--no-maol-warmup-sanity` only for debugging.

### 2. Scale probe

The probe starts from the warm-up checkpoint and runs exactly 100 steps.  By
default it is **measurement-only** (`--maol-probe-no-update`): it never calls
`optimizer.step()`, so it cannot damage the warm-up state.  Use
`--maol-probe-update` to run a clipped 100-step trajectory instead.

```powershell
python train.py `
  --data-dir Dataset `
  --device cuda `
  --batch-size 128 `
  --num-workers 8 `
  --maol-probe-only `
  --maol-probe-steps 100 `
  --maol-probe-no-update `
  --maol-warmup-ckpt checkpoints/maol_warmup.pt `
  --checkpoint checkpoints/maol_model.pt `
  --json-dir runs
```

Outputs:

- `probe_metrics.jsonl`: per-step `loss`, `grad_h`, `grad_param`,
  `total_grad_norm` (pre-clip), `total_grad_norm_postclip`, `tau`, `lr`
- `probe_summary.json`: mean/median/p90/min/max for each metric; per-term
  gradients for `backbone`, `fc1`, `centers`, `radii`

`grad_h` is the gradient norm w.r.t. the shared embedding `h`.  `L_nest`,
`L_cov`, `L_center` have zero `grad_h` by construction; inspect their
`grad_param` (centers/radii) and loss instead.

Lambda guidance:

- If a term's `grad_h` is more than 10x `CE`, lower its lambda to `0.1` or
  `0.01`.
- If a term's `grad_h` is more than 100x smaller than `CE`, raise its lambda to
  `2-5`.
- Target: all six terms within roughly one order of magnitude.
- `L_sep` is noisy early; starting at `0.1` is reasonable.
- `L_cov` and `L_center` are stability terms; keep them in `0.1-1.0`.

### 3. Full MAOL training

```powershell
python train.py `
  --data-dir Dataset `
  --device cuda `
  --epochs 120 `
  --batch-size 128 `
  --model-width-scale 1.5 `
  --maol-mode full `
  --ema 0.9993 `
  --maol-warmup-ckpt checkpoints/maol_warmup.pt `
  --maol-lambda-intra 1.0 `
  --maol-lambda-sep 2.0 `
  --maol-lambda-nest 1.0 `
  --maol-lambda-cov 0.5 `
  --maol-lambda-center 0.5 `
  --checkpoint checkpoints/maol_model.pt `
  --json-dir runs `
  --plot-real-time
```

If `--maol-warmup-ckpt` exists, full training loads it, runs the same sanity
gate on the validation split, and skips warm-up.  A failed gate exits with code
`2` before any training step.  Use `--maol-force-warmup` to regenerate the
checkpoint.  Without the flag, a missing warm-up checkpoint is created
automatically and must pass the same gate before training starts.

### 4. CE linear control (no MAOL)

`--maol-mode none` is the no-MAOL control: raw `fc1+ReLU`, `Linear(256, N)`, CE
only, no warm-up, no probe, no wells.  Use the same epochs/LR/augmentation/EMA as
the MAOL run you are comparing against.

The current dataset has 18 leaf classes, so both MAOL and this control run
18-class.  The historical 93.6% MAOIDL model used **12 classes** and a
token-transformer head; it is not a direct comparison with the current runs.

```powershell
python train.py `
  --data-dir Dataset `
  --device cuda `
  --epochs 120 `
  --batch-size 128 `
  --num-workers 8 `
  --seed 42 `
  --amp-mode auto `
  --optimizer adamw `
  --lr 0.0002 `
  --lr-schedule cosine `
  --min-lr-ratio 0.05 `
  --warmup-epochs 5 `
  --weight-decay 1e-4 `
  --grad-clip 1.0 `
  --augment `
  --rotation 20 `
  --brightness 0.25 `
  --contrast 0.25 `
  --saturation 0.25 `
  --model-width-scale 1.5 `
  --maol-mode none `
  --ema 0.9993 `
  --early-stop `
  --early-stop-metric val_acc `
  --patience 50 `
  --min-delta 0.000001 `
  --checkpoint checkpoints/ce_model.pt `
  --json-dir runs `
  --plot-real-time
```

The run directory is named `...-ce-seed_42`; checkpoints record
`head_type=linear`, `embedding_norm=raw`, and `maol=false`.  `predict.py` accepts
both MAOL and CE checkpoints; the GUI remains MAOL-only.

## Inference

```powershell
python predict.py Dataset/living-matters/animals/cat/0001.jpg `
  --weights checkpoints/maol_model.pt --probabilities --top-k 3 --device cuda
```

`predict.py` accepts both MAOL prototype checkpoints (`embedding_norm=l2`) and
CE linear checkpoints (`head_type=linear`, `embedding_norm=raw`, `maol=false`).
Old MAOIDL token/linear checkpoints and old raw-scale MAOL checkpoints fail with
a clear message; the old architectures are not part of this build.

## GUI

```powershell
python gui.py --checkpoint checkpoints/maol_model.pt
```

The GUI is a "MAOL field station": open a checkpoint, then open an image or
camera.  It shows the nearest leaf well, confidence, top-5 probability bars, tree
size, `tau`, and checkpoint metadata.  It does **not** load old MAOIDL models.

## Metrics and plotting

Each run writes:

```text
runs/<timestamp>-maol-seed_42/   # MAOL
runs/<timestamp>-ce-seed_42/     # CE linear control
  config.json
  warmup_metrics.jsonl      # MAOL only
  warmup_stats.json         # MAOL only
  epoch_metrics.jsonl
  summary.json
```

Per epoch the trainer logs train/val total, CE, intra, sep, nest, cov, center,
accuracy, gap, gradient norm, hidden norm, `tau`, `d_intra`, `d_inter`,
`margin_cap`, mean leaf radius, mean leaf center norm, and the five lambdas.
The CE control logs zero for the well/geometry fields and no `warmup_*` files.

Plotting is owned by the trainer:

```powershell
python train.py ... --plot-once
python train.py ... --plot-real-time
```

`plot.py` can also be run directly to pick an existing JSONL file.

## Training policy notes

- MixUp, CutMix, focal loss, and label smoothing are removed.
- RandAugment remains; validation is always clean.
- EMA is optional and **off by default**.  `--ema` enables it with
  `--ema-decay` (default `0.9993`); `--ema 0.999` sets the decay directly;
  `--no-ema` disables it.  BatchNorm running statistics and MAOL statistic
  buffers are copied exactly from the live model each update; only parameters
  are averaged.  Use the same EMA setting for MAOL and CE controls.
- Class weighting is optional and **off by default** (standard CE).
- Temporary backbone freeze is optional and **off by default**.
- AMP is supported with `--amp-mode {auto,on,off}`.
- Checkpoints are loaded with `weights_only=True`; load only trusted files.
- Warm-up and probe do **not** use `--init-from`; MAOL starts from the current
  backbone and forms the unit-sphere geometry itself.

## Tests

```powershell
$env:MPLBACKEND = "Agg"
.\venv313\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```
