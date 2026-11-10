# 2026-11-04 EMA Buffer Copy and Decay CLI

## Symptom

The EMA MAOL run (`runs/20261102-042306-maol-seed_42`) had identical train
metrics to the non-EMA run (`runs/20261101-014203-maol-seed_42`) but a lower
best val_acc:

```text
EMA:    89.93% @ epoch 89
no EMA: 90.08% @ epoch 112
```

EMA was better across most epochs (mean +0.46 pp) and had a better final
val_loss, but its best-checkpoint metric was lower and macro accuracy dropped
0.32 pp, mostly on mouse/coffee-mug/drum/mushroom.

## Root cause investigated

`ModelEMA.update` blended **all floating state**, including BatchNorm
`running_mean` / `running_var`.  That evaluates EMA parameters with stale,
averaged BN statistics.  The live model's BN statistics describe the current
activations; they should not be averaged with the EMA parameters.

## Changes

### 1. Copy all non-parameter buffers exactly

`ModelEMA` now computes `buffer_keys` from `ema_model.named_buffers()` and
copies those keys from the live model on every update.  This covers:

- BatchNorm `running_mean`, `running_var`, `num_batches_tracked`;
- MAOL `head.c_ema`, `head.q`, `head.tau`, `head.node_weights`, margins,
  separation weights.

Only parameters are EMA-blended.  `skip_prefixes` remains supported for
compatibility.

### 2. `--ema` accepts an optional double

```text
--ema              # enabled, uses --ema-decay (default 0.9993)
--ema 0.999        # enabled, decay = 0.999
--no-ema           # disabled
--ema-decay 0.9993 # fallback decay for bare --ema
```

`_resolve_ema_args` normalizes these into a bool `args.ema` and a validated
`args.ema_decay` before the run config is written.  Invalid values outside
`(0, 1)` exit with a clear message.

## Validation

- 41 tests pass.
- CPU smoke: `--ema 0.999` wrote `ema=true`, `ema_decay=0.999` to config.
- CPU smoke: bare `--ema` wrote `ema=true`, `ema_decay=0.9993`.
- Buffer-copy unit test confirms a non-parameter buffer is copied exactly while
  the parameter is EMA-blended.

## What this does not prove

This change removes a known mismatch; it does not guarantee EMA will beat the
live model on best val_acc.  The next paired run should evaluate both live and
EMA checkpoints at each epoch and compare best, final, and macro accuracy.

## Security notes

- Checkpoint loading still uses `weights_only=True`.
- No new shell execution, network access, or arbitrary deserialization.
- EMA decay is validated to `0 < decay < 1`.

## Correction (2026-11-09)

The buffer-copy decision documented above was wrong. Copying live BatchNorm
statistics into an EMA model whose parameters lag creates a weights/statistics
mismatch, not consistency. The 2026-11-04 MAOL run (`ema_decay=0.9999`, same
code) collapsed from warm-up accuracy `73.28%` at epoch 1 to `44.24%` at epoch
51 while the live training trajectory stayed bit identical to the healthy runs.

Measured on the exact validation split by swapping BatchNorm buffers:

```text
EMA epoch-1 checkpoint as-is                           73.455%
EMA epoch-1 weights + warm-up BatchNorm buffers        63.901%
non-EMA epoch-112 checkpoint as-is                     89.934%
non-EMA epoch-112 weights + warm-up BatchNorm buffers   4.222%
```

`14ecf02` was therefore reverted in `ModelEMA.update` on 2026-11-09: parameters
and floating buffers (BatchNorm running statistics) are EMA-blended with the
same decay again; integer counters and `skip_prefixes` MAOL statistics are
still copied exactly. The `--ema <decay>` CLI from this change is kept.

See `Agent_History/docs/plans/2026-11-09-ema-bn-regression.md` for the full
evidence and fix.
