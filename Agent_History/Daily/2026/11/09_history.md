# 2026-11-09

## Task

Diagnose why the 2026-11-04 MAOL training triple (warm-up, probe, full run)
produced a much worse result than the earlier runs.

## Findings

- The 11-04 warm-up and probe are healthy and geometrically identical to the
  11-01 runs. The main run was a warm-up-accuracy start followed by monotone
  validation collapse: `73.28%` at epoch 1, `66.02%` at epoch 2, `44.24%` at
  epoch 51, early-stopped with `best_epoch=1`.
- Training metrics are bit identical to the 11-01 (non-EMA) and 11-02
  (legacy-EMA) runs, so the live model was not the problem.
- Root cause: commit `14ecf02` changed `ModelEMA` to copy live BatchNorm
  `running_mean`/`running_var` into the EMA model while EMA parameters lag. With
  `ema_decay=0.9999` (effective window about 47 epochs), validation used weights
  and normalization statistics from different training epochs.
- Decisive measurement on the exact validation split: `73.455%` (EMA epoch-1
  checkpoint) dropped to `63.901%` after swapping in warm-up BatchNorm buffers,
  and `89.934%` (epoch-112 non-EMA) dropped to `4.222%`. A one-epoch
  mismatch reproduces the collapsed run's epoch-2 accuracy.

## Changes

- `ModelEMA` now EMA-blends parameters and floating buffers (BatchNorm running
  statistics) with the same decay. Integer counters and `skip_prefixes` MAOL
  statistics are still copied exactly. This restores the pre-`14ecf02`
  semantics that produced the healthy 11-02 EMA run, while keeping the
  `--ema <decay>` CLI.
- Updated `tests/test_training_utils.py` to expect blended floating buffers and
  added an integer-counter copy test.
- Updated `README.md` EMA notes and added
  `Agent_History/docs/plans/2026-11-09-ema-bn-regression.md`; appended a
  correction to `2026-11-04-ema-bn-buffers.md`.

## Validation

- `42 passed` (full pytest suite).
- No training rerun was possible locally: training runs on the remote DGX
  Spark, and the short local reproduction attempt was interrupted by a machine
  restart.

## Next

- Rerun the 18-class MAOL training on the DGX with `--ema 0.9993` (the tested
  decay), using `checkpoints/maol_warmup.pt` if it passes the sanity gate.
- `checkpoints/maol_model.pt` currently holds the collapsed run's EMA epoch-1
  weights; treat it as invalid. `checkpoints/maol_best_model.pt` (non-EMA,
  89.93%) remains the best valid MAOL checkpoint.
- Consider logging live and EMA validation metrics side by side each epoch.
