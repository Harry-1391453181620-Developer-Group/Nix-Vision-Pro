# 2026-11 Summary

## Focus

- Added the 18-class no-MAOL CE linear control.
- Added `--maol-mode {full,none}` and `--no-maol`.
- Enabled EMA consistently in the documented MAOL and CE training commands.
- Fixed EMA evaluation's BatchNorm handling. The first attempt (2026-11-04)
  copied live statistics into the EMA model; that regressed and was corrected
  on 2026-11-09, see below.
- Added `--ema <decay>`.

## Result

- `TorchCNN(..., maol=False)` uses raw `fc1+ReLU` and `Linear(256, N)`; no wells,
  radii, `c_EMA`, `tau`, warm-up, or geometry losses.
- `predict.py` supports both MAOL prototype and CE linear checkpoints.
- GUI remains MAOL-only.
- `ModelEMA` EMA-blends parameters and floating buffers (BatchNorm
  `running_mean`/`running_var`) with the same decay, so the evaluated EMA model
  is self-consistent. Integer counters and MAOL statistic buffers are copied
  exactly.
- `--ema` accepts an optional decay (`--ema 0.999`); bare `--ema` uses
  `--ema-decay` (default `0.9993`).
- 42 tests pass, including linear-checkpoint metadata, EMA buffer blending, and
  EMA decay CLI resolution.

## Corrections

- All MAOL runs are 18-class; the old 93.6% MAOIDL model was 12-class with a
  token-transformer head, so those two numbers are not comparable.
- See `Agent_History/docs/plans/2026-11-02-class-count-correction.md`.
- 2026-11-04 EMA buffer-copy regression: copying live BatchNorm statistics into
  a lagged EMA model collapsed validation accuracy (73.28% at epoch 1 to 44.24%
  at epoch 51, early stop) while the live training trajectory stayed identical
  to the healthy runs. The fix EMA-blends floating buffers with the parameters
  again. Evidence and measurements in
  `Agent_History/docs/plans/2026-11-09-ema-bn-regression.md`.

## Next Work

- Run the 18-class CE linear control with EMA.
- Re-run MAOL with EMA (decay `0.9993`) on the same 18-class tree and compare
  best val accuracy, macro accuracy, and per-class errors.
- Log live and EMA validation metrics side by side each epoch so EMA
  regressions are visible immediately.
- Keep the old 93.6% number out of the direct comparison: that model was
  **12-class** with a token-transformer head, while all MAOL runs are 18-class.
