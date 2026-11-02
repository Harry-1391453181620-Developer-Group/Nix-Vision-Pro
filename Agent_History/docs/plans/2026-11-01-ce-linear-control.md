# 2026-11-01 CE Linear Control (No MAOL)

## Goal

Provide a fair, 18-class **no-MAOL** control for the current CNN backbone:

- same five-stage CNN / SE backbone and width scale;
- raw `ReLU(fc1(flatten(features)))` embedding;
- plain `Linear(256, num_classes)` head;
- CE only, no wells/radii/`c_EMA`/`tau`/MAOL geometry losses;
- no MAOL warm-up and no probe;
- optional EMA, enabled in both control and future MAOL runs.

The old 93.6% checkpoint used **12 classes** and a different (token-transformer)
head, so it is not a direct comparison with the 18-class MAOL runs.  This
control is the correct 18-class baseline: same backbone, same data, same
augmentation, same epochs/LR, same EMA switch.

## Design

### CLI

```text
--maol-mode {full,none}   # default: full
--no-maol                 # alias for --maol-mode none
```

`best_train_commands.txt` uses the explicit `--maol-mode` form.

### No-MAOL model

`TorchCNN(..., maol=False)`:

- no `PrototypeHead`;
- no L2 normalization in `forward_embedding`;
- `fc2 = nn.Linear(embedding_dim, num_classes)`;
- checkpoint metadata:
  - `head_type=linear`
  - `embedding_norm=raw`
  - `maol=false`
  - `theory=CE`
  - `is_ema` / `ema_decay`
- `predict.py` accepts both linear and MAOL checkpoints; the GUI stays MAOL-only.

### Training

A separate `_run_linear_training` path uses:

- `F.cross_entropy` only;
- the same phase/cosine LR, augmentation, grad clip, early stopping, seed, and
  optional EMA as MAOL;
- run tag `ce` (`runs/<timestamp>-ce-seed_42`);
- zero-valued geometry fields in `epoch_metrics.jsonl` so `plot.py` still works;
- summary with `maol_mode=none`, `warmup_stats=null`, zero lambdas.

MAOL-only flags (`--maol-warmup-only`, `--maol-probe-only`,
`--maol-force-warmup`) are rejected in `none` mode.

## Validation

- 38 unit tests pass.
- CPU smoke: 2-epoch 4-class CE linear run with `--ema`; checkpoint metadata
  `head_type=linear`, `maol=false`, `is_ema=true`; `predict.py` loads it.
- CPU smoke: 1-epoch MAOL run with `--ema` from an L2 warm-up checkpoint.
- Existing MAOL checkpoint still predicts.

## Security notes

- Checkpoints continue to load with `weights_only=True`.
- No new shell execution, pickle fallback, network access, or path traversal.
- Linear checkpoints must carry `maol=false`; old MAOIDL linear checkpoints are
  rejected rather than silently loaded.
