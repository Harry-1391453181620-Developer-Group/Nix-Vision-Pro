# 2026-11-04

## Task

Fix EMA evaluation's BatchNorm buffer mismatch and allow `--ema` to set its
decay directly.

## Changes

- `ModelEMA` now copies all non-parameter buffers from the live model on every
  update.  This includes BatchNorm `running_mean`/`running_var` and MAOL
  `c_EMA`/`q`/`tau`/margins.  Only parameters are EMA-blended.
- `--ema` now accepts an optional double:
  - bare `--ema` uses `--ema-decay` (default `0.9993`);
  - `--ema 0.999` sets the decay;
  - `--no-ema` disables EMA.
- Invalid decay values exit with a clear message.
- Updated `README.md`, `best_train_commands.txt`, and added the design note
  `Agent_History/docs/plans/2026-11-04-ema-bn-buffers.md`.

## Validation

- 41 tests pass.
- CPU smoke `--ema 0.999`: config has `ema=true`, `ema_decay=0.999`.
- CPU smoke bare `--ema`: config has `ema=true`, `ema_decay=0.9993`.
- Buffer-copy unit test passes.

## Context

The previous EMA run (`20261102-042306`) had bit-identical training metrics to
the non-EMA run but a lower best val_acc (89.93% vs 90.08%) and lower macro
accuracy.  This change removes the BN-statistics mismatch; it does not by
itself prove EMA will win on best val_acc.  The next paired run should evaluate
both live and EMA checkpoints at each epoch.
